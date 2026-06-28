import torch
from sentence_transformers import CrossEncoder
from loguru import logger
from config import settings
from retrieval.retriever import RetrievedChunk


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    Much more accurate than bi-encoder cosine similarity for final ranking.
    """

    def __init__(self):
        logger.info(f"Loading reranker: {settings.reranker_model}")
        self.model = CrossEncoder(
            settings.reranker_model,
            device=settings.embedding_device,
        )
        logger.info("Reranker ready")

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k_rerank
        if not chunks:
            return []

        # Score all (query, chunk) pairs
        pairs = [(query, chunk.text) for chunk in chunks]
        with torch.no_grad():
            scores = self.model.predict(pairs)

        # Sort by reranker score descending
        scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

        reranked = []
        for score, chunk in scored[:top_k]:
            chunk.score = float(score)  # overwrite with reranker score
            reranked.append(chunk)

        logger.debug(f"Reranked to top {len(reranked)} chunks")
        return reranked


# Singleton
reranker = CrossEncoderReranker()