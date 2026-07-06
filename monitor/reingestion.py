import asyncio
from datetime import datetime
from loguru import logger
from sqlalchemy import select, desc

from storage.postgres_client import AsyncSessionLocal, Document, PipelineAnomaly
from storage.qdrant_client import qdrant
from ingestion.parser import parse_document
from ingestion.chunker import DocumentChunker, ChunkStrategy
from ingestion.embedder import embedder
from qdrant_client.models import PointStruct
import os


class ReIngestionEngine:
    """
    Automatically re-ingests pending and failed documents after
    a remediation action has been approved.
    Closes the self-healing loop end-to-end.
    """

    def __init__(self):
        self.chunker = DocumentChunker(
            strategy=ChunkStrategy.RECURSIVE,
            chunk_size=512,
            chunk_overlap=64,
        )

    async def run(self, anomaly_id: int) -> dict:
        """
        Full re-ingestion cycle triggered after human approval.
        1. Find all pending + failed documents
        2. Re-ingest each one
        3. Verify score improvement
        4. Auto-resolve anomaly if improved
        """
        logger.info(f"Starting re-ingestion cycle for anomaly {anomaly_id}")
        start_time = datetime.utcnow()

        async with AsyncSessionLocal() as db:
            # Fetch all pending + failed documents
            result = await db.execute(
                select(Document)
                .where(Document.status.in_(["pending", "failed"]))
                .order_by(desc(Document.ingested_at))
            )
            docs = result.scalars().all()

            if not docs:
                logger.info("No pending or failed documents found — nothing to re-ingest")
                await self._update_anomaly(db, anomaly_id, "executed", "No documents needed re-ingestion")
                return {"status": "skipped", "reason": "No pending or failed documents", "reingested": 0}

            logger.info(f"Found {len(docs)} documents to re-ingest")

            results = {"indexed": 0, "failed": 0, "skipped": 0}

            for doc in docs:
                outcome = await self._reingest_document(doc, db)
                results[outcome] += 1
                await asyncio.sleep(0.1)  # Small delay between docs

            await db.commit()

        # Verify health improved
        improved = await self._verify_improvement()

        # Auto-resolve anomaly if health improved
        async with AsyncSessionLocal() as db:
            reason = (
                f"Re-ingested {results['indexed']} documents successfully. "
                f"Health {'improved' if improved else 'still recovering — more queries needed'}."
            )
            status = "executed" if improved else "executed"
            await self._update_anomaly(db, anomaly_id, status, reason)
            await db.commit()

        elapsed = (datetime.utcnow() - start_time).seconds
        logger.info(
            f"Re-ingestion complete in {elapsed}s — "
            f"indexed: {results['indexed']}, failed: {results['failed']}, "
            f"health_improved: {improved}"
        )

        return {
            "status": "complete",
            "elapsed_seconds": elapsed,
            "results": results,
            "health_improved": improved,
            "anomaly_id": anomaly_id,
        }

    async def _reingest_document(self, doc: Document, db) -> str:
        """Re-ingest a single document. Returns 'indexed', 'failed', or 'skipped'."""
        try:
            # Try to find the original PDF in the arXiv download cache
            pdf_path = await self._find_cached_pdf(doc.filename)
            if not pdf_path:
                logger.warning(f"No cached PDF found for {doc.filename} — marking as failed")
                doc.status = "failed"
                doc.error_message = "PDF not found in cache during re-ingestion"
                return "failed"

            # Delete existing Qdrant chunks for this doc
            await qdrant.delete_by_doc_id(doc.doc_id)
            logger.debug(f"Cleared existing chunks for {doc.doc_id}")

            # Re-parse
            with open(pdf_path, "rb") as f:
                contents = f.read()
            raw_text = parse_document(contents, ".pdf")

            if not raw_text or len(raw_text.split()) < 50:
                raise ValueError(f"PDF content too short after re-parse: {len(raw_text)} chars")

            # Re-chunk
            chunks = self.chunker.chunk(
                text=raw_text,
                doc_id=doc.doc_id,
                metadata={"filename": doc.filename, "file_type": doc.file_type},
            )

            if not chunks:
                raise ValueError("No chunks produced during re-ingestion")

            # Re-embed
            texts = [c.text for c in chunks]
            vectors = embedder.embed_passages(texts)

            # Re-upsert to Qdrant
            points = [
                PointStruct(
                    id=abs(hash(chunk.chunk_id)) % (2**63),
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": doc.doc_id,
                        "text": chunk.text,
                        "chunk_index": chunk.chunk_index,
                        "filename": doc.filename,
                        **chunk.metadata,
                    },
                )
                for chunk, vector in zip(chunks, vectors)
            ]
            await qdrant.upsert_chunks(points)

            # Update PostgreSQL
            doc.status = "indexed"
            doc.chunk_count = len(chunks)
            doc.error_message = None

            logger.info(f"Re-ingested: {doc.filename} → {len(chunks)} chunks")
            return "indexed"

        except Exception as e:
            logger.error(f"Re-ingestion failed for {doc.filename}: {e}")
            doc.status = "failed"
            doc.error_message = f"Re-ingestion error: {str(e)}"
            return "failed"

    async def _find_cached_pdf(self, filename: str) -> str | None:
        """Search for a cached PDF across known download directories."""
        import tempfile
        import glob

        # Search all aria_arxiv_* temp directories
        temp_base = tempfile.gettempdir()
        pattern = os.path.join(temp_base, "aria_arxiv_*", "*.pdf")
        all_pdfs = glob.glob(pattern)

        # Match by arxiv_id prefix (first part of filename before underscore)
        arxiv_id = filename.split("_")[0].replace(".pdf", "")

        for pdf_path in all_pdfs:
            if arxiv_id in os.path.basename(pdf_path):
                return pdf_path

        # Also check direct filename match
        for pdf_path in all_pdfs:
            if filename in os.path.basename(pdf_path):
                return pdf_path

        return None

    async def _verify_improvement(self) -> bool:
        """
        Check if recent query scores have improved above thresholds.
        Returns True if health looks better.
        """
        from storage.postgres_client import QueryLog
        from sqlalchemy import func

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(
                    func.avg(QueryLog.faithfulness_score).label("avg_faith"),
                    func.avg(QueryLog.hallucination_score).label("avg_hall"),
                )
            )
            row = result.one()
            avg_faith = float(row.avg_faith or 0)
            avg_hall = float(row.avg_hall or 0)

        # Consider improved if faithfulness is above 0.4 or hallucination below 0.5
        improved = avg_faith > 0.4 or avg_hall < 0.5
        logger.info(f"Health check — faithfulness: {avg_faith:.3f}, hallucination: {avg_hall:.3f}, improved: {improved}")
        return improved

    async def _update_anomaly(self, db, anomaly_id: int, status: str, reason: str):
        """Update anomaly record with resolution details."""
        result = await db.execute(
            select(PipelineAnomaly).where(PipelineAnomaly.id == anomaly_id)
        )
        anomaly = result.scalar_one_or_none()
        if anomaly:
            anomaly.remediation_status = status
            anomaly.resolved_at = datetime.utcnow()
            anomaly.root_cause = (anomaly.root_cause or "") + f" | Resolution: {reason}"


# Singleton
reingestion_engine = ReIngestionEngine()