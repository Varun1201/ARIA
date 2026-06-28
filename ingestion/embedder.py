import torch
from sentence_transformers import SentenceTransformer
from loguru import logger
from config import settings


class BGEEmbedder:
    """
    Local BGE-large-en-v1.5 embedder.
    Runs on CUDA (RTX 5080) with batched inference.
    BGE models require prepending 'Represent this sentence: ' for passages,
    and 'Represent this question for searching relevant passages: ' for queries.
    """

    PASSAGE_PREFIX = "Represent this sentence: "
    QUERY_PREFIX = "Represent this question for searching relevant passages: "

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )
        self.model.eval()
        logger.info(f"Embedder ready on device: {settings.embedding_device}")

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks (passages)."""
        prefixed = [self.PASSAGE_PREFIX + t for t in texts]
        with torch.no_grad():
            embeddings = self.model.encode(
                prefixed,
                batch_size=settings.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100,
            )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single user query."""
        prefixed = self.QUERY_PREFIX + query
        with torch.no_grad():
            embedding = self.model.encode(
                prefixed,
                normalize_embeddings=True,
            )
        return embedding.tolist()

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Batch embed multiple queries."""
        prefixed = [self.QUERY_PREFIX + q for q in queries]
        with torch.no_grad():
            embeddings = self.model.encode(
                prefixed,
                batch_size=settings.embedding_batch_size,
                normalize_embeddings=True,
            )
        return embeddings.tolist()


# Singleton — loaded once at startup
embedder = BGEEmbedder()