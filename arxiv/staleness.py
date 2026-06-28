from datetime import datetime, timedelta
from loguru import logger
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from storage.postgres_client import AsyncSessionLocal, Document, PipelineAnomaly


class StalenessDetector:
    """
    Monitors the age of the document corpus.
    Flags when the newest papers are too old relative to the current date.
    Creates anomalies that trigger the self-healing pipeline to fetch new papers.
    """

    def __init__(self, staleness_days: int = 30):
        self.staleness_days = staleness_days  # Flag if newest doc is older than this

    async def check_corpus_staleness(self) -> dict:
        """
        Check if the corpus has gone stale.
        Returns staleness info dict.
        """
        async with AsyncSessionLocal() as db:
            # Get newest indexed document
            result = await db.execute(
                select(Document)
                .where(Document.status == "indexed")
                .order_by(desc(Document.ingested_at))
                .limit(1)
            )
            newest_doc = result.scalar_one_or_none()

            # Get total document count
            count_result = await db.execute(
                select(func.count(Document.id)).where(Document.status == "indexed")
            )
            total_docs = count_result.scalar() or 0

            if not newest_doc or total_docs == 0:
                return {
                    "is_stale": False,
                    "reason": "No documents ingested yet",
                    "total_docs": 0,
                    "newest_doc_age_days": None,
                }

            age_days = (datetime.utcnow() - newest_doc.ingested_at).days
            is_stale = age_days > self.staleness_days

            if is_stale:
                logger.warning(
                    f"Corpus is stale — newest doc is {age_days} days old "
                    f"(threshold: {self.staleness_days} days)"
                )
                await self._create_staleness_anomaly(age_days, total_docs, db)

            return {
                "is_stale": is_stale,
                "newest_doc_age_days": age_days,
                "total_docs": total_docs,
                "staleness_threshold_days": self.staleness_days,
                "newest_doc": newest_doc.filename,
                "reason": f"Newest document is {age_days} days old" if is_stale else "Corpus is fresh",
            }

    async def get_corpus_stats(self) -> dict:
        """Get detailed stats about the current corpus."""
        async with AsyncSessionLocal() as db:
            # Count by status
            status_result = await db.execute(
                select(Document.status, func.count(Document.id).label("count"))
                .group_by(Document.status)
            )
            status_counts = {row.status: int(row.count) for row in status_result.all()}

            # Get oldest and newest
            oldest_result = await db.execute(
                select(Document)
                .where(Document.status == "indexed")
                .order_by(Document.ingested_at)
                .limit(1)
            )
            oldest = oldest_result.scalar_one_or_none()

            newest_result = await db.execute(
                select(Document)
                .where(Document.status == "indexed")
                .order_by(desc(Document.ingested_at))
                .limit(1)
            )
            newest = newest_result.scalar_one_or_none()

            # Total chunks via sum
            chunks_result = await db.execute(
                select(func.sum(Document.chunk_count))
                .where(Document.status == "indexed")
            )
            total_chunks = int(chunks_result.scalar() or 0)

            return {
                "document_counts": status_counts,
                "total_chunks": total_chunks,
                "oldest_document": {
                    "filename": oldest.filename if oldest else None,
                    "ingested_at": oldest.ingested_at.isoformat() if oldest else None,
                },
                "newest_document": {
                    "filename": newest.filename if newest else None,
                    "ingested_at": newest.ingested_at.isoformat() if newest else None,
                },
                "corpus_age_days": (
                    (datetime.utcnow() - oldest.ingested_at).days
                    if oldest else None
                ),
            }

    async def _create_staleness_anomaly(
        self,
        age_days: int,
        total_docs: int,
        db: AsyncSession,
    ):
        """Create a staleness anomaly to trigger self-healing."""
        # Check for recent duplicate
        cutoff = datetime.utcnow() - timedelta(hours=24)
        existing = await db.execute(
            select(PipelineAnomaly).where(
                PipelineAnomaly.anomaly_type == "corpus_staleness",
                PipelineAnomaly.detected_at > cutoff,
            )
        )
        if existing.scalar_one_or_none():
            return  # Already flagged today

        anomaly = PipelineAnomaly(
            anomaly_type="corpus_staleness",
            severity="medium" if age_days < 60 else "high",
            description=f"Research corpus is {age_days} days old with {total_docs} documents — new papers may be available",
            extra_metadata={
                "age_days": age_days,
                "total_docs": total_docs,
                "suggested_action": "fetch_new_papers",
            },
        )
        db.add(anomaly)
        await db.commit()
        logger.info(f"Created corpus_staleness anomaly (age: {age_days} days)")


# Singleton
staleness_detector = StalenessDetector(staleness_days=30)