from typing import Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
)
from loguru import logger
from config import settings


class ARIAQdrantClient:
    def __init__(self):
        self.client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.collection = settings.qdrant_collection

    async def ensure_collection(self):
        """Create collection if it doesn't exist."""
        exists = await self.client.collection_exists(self.collection)
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=settings.qdrant_vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection}")
        else:
            logger.info(f"Collection already exists: {self.collection}")

    async def upsert_chunks(self, points: list[PointStruct]):
        """Insert or update chunk vectors."""
        await self.client.upsert(
            collection_name=self.collection,
            points=points,
        )
        logger.debug(f"Upserted {len(points)} chunks to Qdrant")

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        doc_id_filter: Optional[str] = None,
    ) -> list[ScoredPoint]:
        """Dense vector search with optional doc filter."""
        query_filter = None
        if doc_id_filter:
            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))]
            )

        results = await self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return results

    async def delete_by_doc_id(self, doc_id: str):
        """Remove all chunks belonging to a document."""
        from qdrant_client.models import FilterSelector
        await self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            ),
        )
        logger.info(f"Deleted all chunks for doc_id: {doc_id}")

    async def get_collection_info(self) -> dict:
        info = await self.client.get_collection(self.collection)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }


# Singleton instance
qdrant = ARIAQdrantClient()