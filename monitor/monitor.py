import asyncio
from datetime import datetime, timedelta
from loguru import logger
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from storage.postgres_client import AsyncSessionLocal, QueryLog, Document, PipelineAnomaly
from monitor.anomaly_detector import anomaly_detector, AnomalySignal
from config import settings


class PipelineMonitor:
    """
    Async monitor that runs in the background.
    Periodically fetches recent metrics from PostgreSQL
    and runs anomaly detection.
    """

    def __init__(self, interval_seconds: int = 60):
        self.interval_seconds = interval_seconds
        self.running = False

    async def start(self):
        """Start the monitor loop."""
        self.running = True
        logger.info(f"Pipeline monitor started (interval: {self.interval_seconds}s)")
        while self.running:
            try:
                await self._run_checks()
            except Exception as e:
                logger.error(f"Monitor check failed: {e}")
            await asyncio.sleep(self.interval_seconds)

    def stop(self):
        self.running = False
        logger.info("Pipeline monitor stopped")

    async def _run_checks(self):
        async with AsyncSessionLocal() as db:
            # Fetch recent query logs
            query_result = await db.execute(
                select(QueryLog)
                .order_by(desc(QueryLog.created_at))
                .limit(settings.anomaly_rolling_window)
            )
            recent_queries = query_result.scalars().all()

            # Fetch recent document statuses
            doc_result = await db.execute(
                select(Document.status)
                .order_by(desc(Document.ingested_at))
                .limit(20)
            )
            recent_doc_statuses = [row[0] for row in doc_result.all()]

            if not recent_queries:
                logger.debug("No queries yet — skipping anomaly check")
                return

            # Extract metric series
            faithfulness_scores = [
                q.faithfulness_score for q in recent_queries
                if q.faithfulness_score is not None
            ]
            relevance_scores = [
                q.relevance_score for q in recent_queries
                if q.relevance_score is not None
            ]
            hallucination_flags = [
                q.is_hallucination_flagged for q in recent_queries
                if q.is_hallucination_flagged is not None
            ]
            latencies = [
                q.latency_ms for q in recent_queries
                if q.latency_ms is not None
            ]

            # Run detectors
            signals = anomaly_detector.run_all(
                faithfulness_scores=faithfulness_scores,
                relevance_scores=relevance_scores,
                hallucination_flags=hallucination_flags,
                ingestion_statuses=recent_doc_statuses,
                latencies_ms=latencies,
            )

            # Persist anomalies to DB
            for signal in signals:
                if not await self._is_duplicate(db, signal):
                    anomaly = PipelineAnomaly(
                        anomaly_type=signal.anomaly_type,
                        severity=signal.severity,
                        description=signal.description,
                        extra_metadata={
                            "metric_value": signal.metric_value,
                            "threshold_value": signal.threshold_value,
                            **(signal.extra or {}),
                        },
                    )
                    db.add(anomaly)

            await db.commit()

    async def _is_duplicate(self, db: AsyncSession, signal: AnomalySignal) -> bool:
        """Avoid creating duplicate anomalies within 10 minutes."""
        cutoff = datetime.utcnow() - timedelta(minutes=10)
        result = await db.execute(
            select(PipelineAnomaly)
            .where(
                PipelineAnomaly.anomaly_type == signal.anomaly_type,
                PipelineAnomaly.detected_at > cutoff,
                PipelineAnomaly.remediation_status == "pending",
            )
        )
        return result.scalar_one_or_none() is not None

    async def get_active_anomalies(self) -> list[dict]:
        """Return all unresolved anomalies."""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(PipelineAnomaly)
                .where(PipelineAnomaly.remediation_status.in_(["pending", "approved"]))
                .order_by(desc(PipelineAnomaly.detected_at))
            )
            anomalies = result.scalars().all()
            return [
                {
                    "id": a.id,
                    "anomaly_type": a.anomaly_type,
                    "severity": a.severity,
                    "description": a.description,
                    "remediation_status": a.remediation_status,
                    "detected_at": a.detected_at.isoformat(),
                    "extra_metadata": a.extra_metadata,
                }
                for a in anomalies
            ]


# Singleton
pipeline_monitor = PipelineMonitor(interval_seconds=60)