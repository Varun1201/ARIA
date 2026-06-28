import uuid
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from loguru import logger
from groq import Groq

from config import settings
from storage.postgres_client import get_db, QueryLog
from retrieval.retriever import DenseRetriever
from retrieval.reranker import reranker
from auditor.auditor import auditor

groq_client = Groq(api_key=settings.groq_api_key)

router = APIRouter(prefix="/query", tags=["Query"])
retriever = DenseRetriever()


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    doc_id_filter: Optional[str] = None


class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    score: float


class AuditSummary(BaseModel):
    faithfulness_score: float
    relevance_score: float
    hallucination_score: float
    is_hallucination_flagged: bool
    groq_verdict: Optional[str] = None
    groq_reason: Optional[str] = None
    citations: list[dict]


class QueryResponse(BaseModel):
    query_id: str
    query: str
    answer: str
    source_chunks: list[SourceChunk]
    audit: AuditSummary
    latency_ms: int


# ── Groq LLM Call ────────────────────────────────────────────────────────────

async def call_groq(prompt: str) -> str:
    import asyncio
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.groq_temperature,
            max_tokens=1024,
        )
    )
    return response.choices[0].message.content.strip()


def build_rag_prompt(query: str, chunks: list) -> str:
    context = "\n\n".join(
        f"[Source {i+1} | {c.filename}]\n{c.text}"
        for i, c in enumerate(chunks)
    )
    return f"""You are ARIA, an intelligent RAG assistant. Answer the question using ONLY the provided context.
If the answer cannot be found in the context, say "I cannot answer this based on the provided documents."
Do NOT make up information. Cite the source number (e.g. [Source 1]) when referencing specific facts.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


# ── Background audit task ─────────────────────────────────────────────────────

async def _run_audit(
    query_id: str,
    query: str,
    answer: str,
    chunk_texts: list[str],
    chunk_ids: list[str],
    db: AsyncSession,
):
    """Run audit in background and update QueryLog with scores."""
    try:
        result = await auditor.audit(
            query_id=query_id,
            query=query,
            answer=answer,
            chunks=chunk_texts,
            chunk_ids=chunk_ids,
        )

        # Update QueryLog with audit scores
        await db.execute(
            update(QueryLog)
            .where(QueryLog.query_id == query_id)
            .values(
                faithfulness_score=result.faithfulness_score,
                relevance_score=result.relevance_score,
                hallucination_score=result.hallucination_score,
                is_hallucination_flagged=result.is_hallucination_flagged,
            )
        )
        await db.commit()
        logger.info(f"Audit complete for {query_id} — flagged: {result.is_hallucination_flagged}")

    except Exception as e:
        logger.error(f"Audit failed for {query_id}: {e}")


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Submit a question to ARIA's RAG pipeline with full auditing."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start_time = time.time()
    query_id = str(uuid.uuid4())

    # 1. Retrieve
    chunks = await retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        doc_id_filter=request.doc_id_filter,
    )
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Please ingest documents first.",
        )

    # 2. Rerank
    top_chunks = reranker.rerank(query=request.query, chunks=chunks)

    # 3. Generate answer via Groq
    prompt = build_rag_prompt(request.query, top_chunks)
    try:
        answer = await call_groq(prompt)
    except Exception as e:
        logger.error(f"Groq call failed: {e}")
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")

    latency_ms = int((time.time() - start_time) * 1000)

    # 4. Run audit synchronously (so we return scores in the response)
    chunk_texts = [c.text for c in top_chunks]
    chunk_ids = [c.chunk_id for c in top_chunks]

    audit_result = await auditor.audit(
        query_id=query_id,
        query=request.query,
        answer=answer,
        chunks=chunk_texts,
        chunk_ids=chunk_ids,
    )

    # 5. Log everything to PostgreSQL
    log = QueryLog(
        query_id=query_id,
        query_text=request.query,
        response_text=answer,
        retrieved_chunk_ids=chunk_ids,
        faithfulness_score=audit_result.faithfulness_score,
        relevance_score=audit_result.relevance_score,
        hallucination_score=audit_result.hallucination_score,
        is_hallucination_flagged=audit_result.is_hallucination_flagged,
        latency_ms=latency_ms,
    )
    db.add(log)
    await db.commit()

    logger.info(
        f"Query {query_id} — latency: {latency_ms}ms | "
        f"faithful: {audit_result.faithfulness_score:.2f} | "
        f"flagged: {audit_result.is_hallucination_flagged}"
    )

    return QueryResponse(
        query_id=query_id,
        query=request.query,
        answer=answer,
        source_chunks=[
            SourceChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                filename=c.filename,
                text=c.text,
                score=c.score,
            )
            for c in top_chunks
        ],
        audit=AuditSummary(
            faithfulness_score=audit_result.faithfulness_score,
            relevance_score=audit_result.relevance_score,
            hallucination_score=audit_result.hallucination_score,
            is_hallucination_flagged=audit_result.is_hallucination_flagged,
            groq_verdict=audit_result.groq_verdict,
            groq_reason=audit_result.groq_reason,
            citations=audit_result.citations,
        ),
        latency_ms=latency_ms,
    )


@router.get("/audit/{query_id}", tags=["Audit"])
async def get_audit(query_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve audit scores for a past query."""
    result = await db.execute(select(QueryLog).where(QueryLog.query_id == query_id))
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=404, detail=f"Query {query_id} not found")
    return {
        "query_id": log.query_id,
        "query_text": log.query_text,
        "faithfulness_score": log.faithfulness_score,
        "relevance_score": log.relevance_score,
        "hallucination_score": log.hallucination_score,
        "is_hallucination_flagged": log.is_hallucination_flagged,
        "latency_ms": log.latency_ms,
        "created_at": log.created_at.isoformat(),
    }