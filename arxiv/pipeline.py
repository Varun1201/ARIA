import os
import uuid
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from arxiv.fetcher import arxiv_fetcher, ArxivPaper, TOPIC_CLUSTERS
from arxiv.downloader import pdf_downloader
from storage.postgres_client import AsyncSessionLocal, Document
from storage.qdrant_client import qdrant
from ingestion.parser import parse_document
from ingestion.chunker import DocumentChunker, ChunkStrategy
from ingestion.embedder import embedder
from qdrant_client.models import PointStruct


class ArxivIngestionPipeline:
    """
    End-to-end pipeline:
    arXiv API → PDF download → parse → chunk → embed → Qdrant + PostgreSQL
    """

    def __init__(self):
        self.chunker = DocumentChunker(
            strategy=ChunkStrategy.RECURSIVE,
            chunk_size=400,   # Smaller chunks for dense research text
            chunk_overlap=80,
        )

    async def ingest_paper(
        self,
        paper: ArxivPaper,
        pdf_path: str,
        db: AsyncSession,
    ) -> dict:
        """Ingest a single paper into ARIA."""

        # Check if already ingested
        existing = await db.execute(
            select(Document).where(
                Document.filename == f"{paper.arxiv_id}.pdf"
            )
        )
        if existing.scalar_one_or_none():
            logger.debug(f"Already ingested: {paper.arxiv_id}")
            return {"status": "skipped", "arxiv_id": paper.arxiv_id}

        doc_id = str(uuid.uuid4())

        try:
            # 1. Parse PDF
            with open(pdf_path, "rb") as f:
                contents = f.read()
            raw_text = parse_document(contents, ".pdf")

            if not raw_text or len(raw_text.split()) < 50:
                raise ValueError(f"PDF too short or empty: {len(raw_text)} chars")

            # 2. Prepend metadata for richer retrieval
            metadata_header = f"""
Title: {paper.title}
Authors: {', '.join(paper.authors)}
Published: {paper.published.strftime('%Y-%m-%d')}
Topic: {paper.topic_cluster}
Abstract: {paper.abstract}

---

"""
            full_text = metadata_header + raw_text

            # 3. Chunk
            chunks = self.chunker.chunk(
                text=full_text,
                doc_id=doc_id,
                metadata={
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": paper.authors,
                    "published": paper.published.isoformat(),
                    "topic_cluster": paper.topic_cluster,
                    "source": "arxiv",
                },
            )

            if not chunks:
                raise ValueError("No chunks produced")

            # 4. Embed
            texts = [c.text for c in chunks]
            vectors = embedder.embed_passages(texts)

            # 5. Upsert to Qdrant
            points = [
                PointStruct(
                    id=abs(hash(chunk.chunk_id)) % (2**63),
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_id": doc_id,
                        "text": chunk.text,
                        "chunk_index": chunk.chunk_index,
                        "filename": f"{paper.arxiv_id}.pdf",
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "topic_cluster": paper.topic_cluster,
                        "published": paper.published.isoformat(),
                        **chunk.metadata,
                    },
                )
                for chunk, vector in zip(chunks, vectors)
            ]
            await qdrant.upsert_chunks(points)

            # 6. Save to PostgreSQL
            doc_record = Document(
                doc_id=doc_id,
                filename=f"{paper.arxiv_id}.pdf",
                file_type=".pdf",
                chunk_count=len(chunks),
                status="indexed",
            )
            db.add(doc_record)
            await db.commit()

            logger.info(
                f"Ingested: {paper.title[:50]}... "
                f"({len(chunks)} chunks, {paper.topic_cluster})"
            )
            return {
                "status": "indexed",
                "arxiv_id": paper.arxiv_id,
                "doc_id": doc_id,
                "chunk_count": len(chunks),
                "title": paper.title,
            }

        except Exception as e:
            logger.error(f"Failed to ingest {paper.arxiv_id}: {e}")
            doc_record = Document(
                doc_id=doc_id,
                filename=f"{paper.arxiv_id}.pdf",
                file_type=".pdf",
                status="failed",
                error_message=str(e),
            )
            db.add(doc_record)
            await db.commit()
            return {"status": "failed", "arxiv_id": paper.arxiv_id, "error": str(e)}

    async def run_cluster(
        self,
        cluster_name: str,
        max_per_query: int = 3,
    ) -> list[dict]:
        """Fetch and ingest all papers from a topic cluster."""
        logger.info(f"Starting ingestion for cluster: {cluster_name}")

        # Fetch paper metadata
        papers = await arxiv_fetcher.fetch_cluster(cluster_name, max_per_query)
        if not papers:
            return []

        # Download PDFs
        downloaded = await pdf_downloader.download_papers(papers, max_concurrent=2)

        # Ingest each paper
        results = []
        async with AsyncSessionLocal() as db:
            for paper, pdf_path in downloaded:
                result = await self.ingest_paper(paper, pdf_path, db)
                results.append(result)

        indexed = sum(1 for r in results if r["status"] == "indexed")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        failed = sum(1 for r in results if r["status"] == "failed")

        logger.info(
            f"Cluster '{cluster_name}' complete — "
            f"indexed: {indexed}, skipped: {skipped}, failed: {failed}"
        )
        return results

    async def run_all_clusters(self, max_per_query: int = 2) -> dict:
        """Run ingestion across all topic clusters."""
        all_results = {}
        for cluster in TOPIC_CLUSTERS:
            results = await self.run_cluster(cluster, max_per_query)
            all_results[cluster] = results
        return all_results

    async def run_recent(self, days: int = 7) -> list[dict]:
        """Fetch and ingest papers published in the last N days."""
        papers = await arxiv_fetcher.fetch_recent(days=days)
        if not papers:
            logger.info("No recent papers found")
            return []

        downloaded = await pdf_downloader.download_papers(papers, max_concurrent=2)
        results = []
        async with AsyncSessionLocal() as db:
            for paper, pdf_path in downloaded:
                result = await self.ingest_paper(paper, pdf_path, db)
                results.append(result)

        return results


# Singleton
arxiv_pipeline = ArxivIngestionPipeline()