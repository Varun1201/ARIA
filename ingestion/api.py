import uuid
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from config import settings
from storage.postgres_client import get_db, Document
from storage.qdrant_client import qdrant
from ingestion.chunker import DocumentChunker, ChunkStrategy
from ingestion.embedder import embedder
from ingestion.parser import parse_document
from qdrant_client.models import PointStruct

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    message: str


class DocumentStatus(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: int
    ingested_at: str
    error_message: Optional[str] = None


# ── Background ingestion task ─────────────────────────────────────────────────

async def _process_document(
    doc_id: str,
    filename: str,
    file_type: str,
    raw_text: str,
    db: AsyncSession,
):
    """Full ingestion pipeline: chunk → embed → store in Qdrant + PostgreSQL."""
    try:
        # 1. Chunk
        chunker = DocumentChunker(strategy=ChunkStrategy.RECURSIVE)
        chunks = chunker.chunk(
            text=raw_text,
            doc_id=doc_id,
            metadata={"filename": filename, "file_type": file_type},
        )

        if not chunks:
            raise ValueError("No chunks produced — document may be empty")

        # 2. Embed
        texts = [c.text for c in chunks]
        vectors = embedder.embed_passages(texts)

        # 3. Upsert to Qdrant
        points = [
            PointStruct(
                id=abs(hash(chunk.chunk_id)) % (2**63),  # deterministic int ID
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "filename": filename,
                    **chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        await qdrant.upsert_chunks(points)

        # 4. Update PostgreSQL
        from sqlalchemy import select
        result = await db.execute(select(Document).where(Document.doc_id == doc_id))
        doc_record = result.scalar_one_or_none()
        if doc_record:
            doc_record.status = "indexed"
            doc_record.chunk_count = len(chunks)
            await db.commit()

        logger.info(f"Successfully ingested {filename} → {len(chunks)} chunks")

    except Exception as e:
        logger.error(f"Ingestion failed for {doc_id}: {e}")
        result = await db.execute(select(Document).where(Document.doc_id == doc_id))
        doc_record = result.scalar_one_or_none()
        if doc_record:
            doc_record.status = "failed"
            doc_record.error_message = str(e)
            await db.commit()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=IngestResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document (PDF, DOCX, TXT) for ingestion into ARIA."""
    allowed_types = {".pdf", ".docx", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_types}",
        )

    # Read file bytes
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    # Parse to raw text
    try:
        raw_text = parse_document(contents, suffix)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse document: {e}")

    # Create DB record
    doc_id = str(uuid.uuid4())
    doc_record = Document(
        doc_id=doc_id,
        filename=file.filename,
        file_type=suffix,
        status="pending",
    )
    db.add(doc_record)
    await db.commit()

    # Queue background ingestion
    background_tasks.add_task(
        _process_document,
        doc_id=doc_id,
        filename=file.filename,
        file_type=suffix,
        raw_text=raw_text,
        db=db,
    )

    return IngestResponse(
        doc_id=doc_id,
        filename=file.filename,
        status="pending",
        message="Document queued for ingestion. Poll /ingest/status/{doc_id} for updates.",
    )


@router.get("/status/{doc_id}", response_model=DocumentStatus)
async def get_document_status(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Check the ingestion status of a document."""
    from sqlalchemy import select
    result = await db.execute(select(Document).where(Document.doc_id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    return DocumentStatus(
        doc_id=doc.doc_id,
        filename=doc.filename,
        status=doc.status,
        chunk_count=doc.chunk_count,
        ingested_at=doc.ingested_at.isoformat(),
        error_message=doc.error_message,
    )


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Remove a document from both Qdrant and PostgreSQL."""
    from sqlalchemy import select, delete
    result = await db.execute(select(Document).where(Document.doc_id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    await qdrant.delete_by_doc_id(doc_id)
    await db.execute(delete(Document).where(Document.doc_id == doc_id))
    await db.commit()

    return {"message": f"Document {doc_id} deleted successfully"}