import numpy as np
from loguru import logger
from qdrant_client.models import Filter, FieldCondition, MatchValue
from storage.qdrant_client import qdrant
from config import settings


class DriftWatchdog:
    """
    Monitors embedding drift in the document corpus.
    Computes the corpus centroid and checks new documents
    against it to detect distribution shift.
    """

    def __init__(self):
        self._centroid: list[float] | None = None
        self._centroid_doc_count: int = 0

    async def compute_centroid(self) -> list[float] | None:
        """
        Compute the mean embedding vector of all chunks in Qdrant.
        This is the 'expected' distribution center.
        """
        try:
            results, _ = await qdrant.client.scroll(
                collection_name=qdrant.collection,
                limit=500,
                with_vectors=True,
                with_payload=False,
            )

            if not results:
                logger.debug("No vectors in collection yet — skipping centroid computation")
                return None

            vectors = np.array([point.vector for point in results])
            centroid = vectors.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)  # normalize

            self._centroid = centroid.tolist()
            self._centroid_doc_count = len(results)
            logger.debug(f"Computed centroid from {len(results)} chunks")
            return self._centroid

        except Exception as e:
            logger.error(f"Centroid computation failed: {e}")
            return None

    async def score_document_drift(self, doc_id: str) -> float | None:
        """
        Compute how far a specific document's chunks are from the corpus centroid.
        Returns cosine distance (0 = identical, 1 = completely different).
        """
        if self._centroid is None:
            await self.compute_centroid()
        if self._centroid is None:
            return None

        try:
            results, _ = await qdrant.client.scroll(
                collection_name=qdrant.collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
                limit=100,
                with_vectors=True,
                with_payload=False,
            )

            if not results:
                return None

            doc_vectors = np.array([point.vector for point in results])
            doc_centroid = doc_vectors.mean(axis=0)
            doc_centroid = doc_centroid / np.linalg.norm(doc_centroid)

            corpus_centroid = np.array(self._centroid)
            cosine_similarity = float(np.dot(doc_centroid, corpus_centroid))
            cosine_distance = 1.0 - cosine_similarity

            logger.debug(f"Doc {doc_id} drift score: {cosine_distance:.4f}")
            return round(cosine_distance, 4)

        except Exception as e:
            logger.error(f"Drift scoring failed for {doc_id}: {e}")
            return None

    async def check_corpus_drift(self) -> dict:
        """
        Full drift check — recompute centroid and score recent documents.
        Returns drift report.
        """
        centroid = await self.compute_centroid()
        if centroid is None:
            return {"status": "insufficient_data", "drift_scores": {}}

        # Get recent doc IDs from Qdrant payload
        try:
            results, _ = await qdrant.client.scroll(
                collection_name=qdrant.collection,
                limit=200,
                with_vectors=False,
                with_payload=True,
            )

            # Get unique doc_ids
            doc_ids = list({
                point.payload.get("doc_id")
                for point in results
                if point.payload.get("doc_id")
            })

            drift_scores = {}
            high_drift_docs = []

            for doc_id in doc_ids:
                score = await self.score_document_drift(doc_id)
                if score is not None:
                    drift_scores[doc_id] = score
                    if score > settings.drift_alert_threshold:
                        high_drift_docs.append(doc_id)
                        logger.warning(
                            f"High drift detected for doc {doc_id}: {score:.4f} "
                            f"(threshold: {settings.drift_alert_threshold})"
                        )

            avg_drift = sum(drift_scores.values()) / len(drift_scores) if drift_scores else 0

            return {
                "status": "critical" if high_drift_docs else "normal",
                "avg_drift_score": round(avg_drift, 4),
                "drift_threshold": settings.drift_alert_threshold,
                "total_docs_checked": len(doc_ids),
                "high_drift_docs": high_drift_docs,
                "drift_scores": drift_scores,
                "corpus_size": self._centroid_doc_count,
            }

        except Exception as e:
            logger.error(f"Corpus drift check failed: {e}")
            return {"status": "error", "error": str(e)}


# Singleton
drift_watchdog = DriftWatchdog()