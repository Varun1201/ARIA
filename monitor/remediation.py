from loguru import logger
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from storage.postgres_client import AsyncSessionLocal, Document, PipelineAnomaly
from storage.qdrant_client import qdrant
from config import settings


# Actions that require human approval before execution
HIGH_RISK_ACTIONS = {"rebuild_index", "purge_low_quality"}


class RemediationEngine:
    """
    Executes remediation actions to fix detected pipeline anomalies.
    High-risk actions are gated behind human approval.
    """

    async def execute(
        self,
        anomaly_id: int,
        action: str,
        risk_level: str,
        approved: bool = False,
    ) -> dict:
        """
        Execute a remediation action.
        Returns result dict with status and details.
        """
        # Gate high-risk actions
        if action in HIGH_RISK_ACTIONS and not approved:
            logger.warning(f"Action '{action}' requires human approval — setting to pending")
            await self._update_anomaly_status(anomaly_id, "pending", action)
            return {
                "status": "awaiting_approval",
                "action": action,
                "message": f"Action '{action}' is high-risk and requires human approval via POST /monitor/anomalies/{anomaly_id}/approve",
            }

        logger.info(f"Executing remediation: {action} for anomaly {anomaly_id}")

        if action == "re_chunk":
            result = await self._re_chunk_failed_documents()
        elif action == "purge_low_quality":
            result = await self._purge_low_quality_chunks()
        elif action == "rebuild_index":
            result = await self._rebuild_index()
        elif action == "alert_only":
            result = {"status": "alerted", "message": "Anomaly logged for human review"}
        else:
            result = {"status": "unknown_action", "message": f"Unknown action: {action}"}

        await self._update_anomaly_status(anomaly_id, "executed", action)
        return result

    # ── Remediation Actions ───────────────────────────────────────────────────

    async def _re_chunk_failed_documents(self) -> dict:
        """Re-process documents that failed or produced poor quality chunks."""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Document)
                .where(Document.status.in_(["failed", "indexed"]))
                .where(Document.chunk_count <= 1)
                .order_by(desc(Document.ingested_at))
                .limit(5)
            )
            docs = result.scalars().all()

            if not docs:
                return {"status": "nothing_to_rechunk", "message": "No poor-quality documents found"}

            reprocessed = []
            for doc in docs:
                try:
                    # Mark as pending to trigger re-ingestion
                    doc.status = "pending"
                    doc.chunk_count = 0
                    reprocessed.append(doc.doc_id)
                    logger.info(f"Marked {doc.doc_id} ({doc.filename}) for re-chunking")
                except Exception as e:
                    logger.error(f"Failed to mark {doc.doc_id} for re-chunking: {e}")

            await db.commit()

            return {
                "status": "success",
                "message": f"Marked {len(reprocessed)} documents for re-chunking",
                "doc_ids": reprocessed,
            }

    async def _purge_low_quality_chunks(self) -> dict:
        """Remove chunks with very short text (likely noise)."""
        try:
            # Search for chunks with very short text via scroll
            results, _ = await qdrant.client.scroll(
                collection_name=qdrant.collection,
                limit=500,
                with_payload=True,
                with_vectors=False,
            )

            low_quality_ids = []
            for point in results:
                text = point.payload.get("text", "")
                if len(text.split()) < 10:  # Fewer than 10 words = low quality
                    low_quality_ids.append(point.id)

            if not low_quality_ids:
                return {"status": "nothing_to_purge", "message": "No low-quality chunks found"}

            from qdrant_client.models import PointIdsList
            await qdrant.client.delete(
                collection_name=qdrant.collection,
                points_selector=PointIdsList(points=low_quality_ids),
            )

            logger.info(f"Purged {len(low_quality_ids)} low-quality chunks")
            return {
                "status": "success",
                "message": f"Purged {len(low_quality_ids)} low-quality chunks",
                "purged_count": len(low_quality_ids),
            }

        except Exception as e:
            logger.error(f"Purge failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _rebuild_index(self) -> dict:
        """
        Nuclear option: delete and rebuild the entire Qdrant collection.
        Re-triggers ingestion for all indexed documents.
        """
        try:
            # Delete collection
            await qdrant.client.delete_collection(qdrant.collection)
            logger.warning("Deleted Qdrant collection for rebuild")

            # Recreate collection
            await qdrant.ensure_collection()
            logger.info("Recreated Qdrant collection")

            # Mark all indexed documents for re-ingestion
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Document).where(Document.status == "indexed")
                )
                docs = result.scalars().all()
                for doc in docs:
                    doc.status = "pending"
                    doc.chunk_count = 0
                await db.commit()
                logger.info(f"Marked {len(docs)} documents for re-ingestion")

            return {
                "status": "success",
                "message": f"Index rebuilt. {len(docs)} documents queued for re-ingestion.",
            }

        except Exception as e:
            logger.error(f"Rebuild failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _update_anomaly_status(
        self, anomaly_id: int, status: str, action: str
    ):
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(PipelineAnomaly).where(PipelineAnomaly.id == anomaly_id)
            )
            anomaly = result.scalar_one_or_none()
            if anomaly:
                anomaly.remediation_status = status
                anomaly.remediation_action = action
                await db.commit()


# Singleton
remediation_engine = RemediationEngine()