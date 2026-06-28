import re
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger
from config import settings


class CitationTracer:
    """
    Maps each sentence in the answer back to the most relevant source chunk.
    Uses the same BGE embedder for semantic similarity.
    """

    def __init__(self):
        # Reuse BGE model already loaded for embeddings
        logger.info("Citation tracer using BGE embeddings")
        self.model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )
        self.model.eval()

    def trace(self, answer: str, chunks: list[str], chunk_ids: list[str]) -> list[dict]:
        """
        Returns a list of citation mappings:
        [
          {
            "sentence": "...",
            "best_chunk_id": "...",
            "best_chunk_index": 0,
            "similarity": 0.87
          },
          ...
        ]
        """
        sentences = self._split_sentences(answer)
        if not sentences or not chunks:
            return []

        with torch.no_grad():
            sentence_embeddings = self.model.encode(
                sentences, normalize_embeddings=True
            )
            chunk_embeddings = self.model.encode(
                chunks, normalize_embeddings=True
            )

        # Cosine similarity matrix: (n_sentences x n_chunks)
        sim_matrix = sentence_embeddings @ chunk_embeddings.T

        citations = []
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < 4:
                continue
            best_chunk_idx = int(sim_matrix[i].argmax())
            best_similarity = float(sim_matrix[i][best_chunk_idx])
            citations.append({
                "sentence": sentence,
                "best_chunk_id": chunk_ids[best_chunk_idx] if chunk_ids else str(best_chunk_idx),
                "best_chunk_index": best_chunk_idx,
                "similarity": round(best_similarity, 4),
            })

        return citations

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]


# Singleton
citation_tracer = CitationTracer()