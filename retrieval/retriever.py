from dataclasses import dataclass
from typing import Optional
from loguru import logger

from storage.qdrant_client import qdrant
from ingestion.embedder import embedder
from config import settings


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    filename: str
    chunk_index: int
    score: float


class DenseRetriever:
    """Retrieves top-K chunks from Qdrant using BGE embeddings."""

    async def retrieve(
        self,
        query: str,
        top_k: int = None,
        doc_id_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k_retrieval
        query_vector = embedder.embed_query(query)

        results = await qdrant.search(
            query_vector=query_vector,
            top_k=top_k,
            doc_id_filter=doc_id_filter,
        )

        chunks = []
        for r in results:
            payload = r.payload or {}
            chunks.append(RetrievedChunk(
                chunk_id=payload.get("chunk_id", ""),
                doc_id=payload.get("doc_id", ""),
                text=payload.get("text", ""),
                filename=payload.get("filename", ""),
                chunk_index=payload.get("chunk_index", 0),
                score=r.score,
            ))

        logger.debug(f"Retrieved {len(chunks)} chunks for query: '{query[:60]}...'")
        return chunks