from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from loguru import logger

from storage.postgres_client import get_db, QueryLog, Document, PipelineAnomaly
from monitor.drift_watchdog import drift_watchdog

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/metrics")
async def get_dashboard_metrics(db: AsyncSession = Depends(get_db)):
    """
    Single endpoint that returns all metrics needed by the dashboard.
    Designed for polling every 30 seconds.
    """

    # ── Overall stats ─────────────────────────────────────────────────────────
    stats_result = await db.execute(
        select(
            func.count(QueryLog.id).label("total_queries"),
            func.avg(QueryLog.faithfulness_score).label("avg_faithfulness"),
            func.avg(QueryLog.relevance_score).label("avg_relevance"),
            func.avg(QueryLog.hallucination_score).label("avg_hallucination"),
            func.count(QueryLog.id).filter(QueryLog.is_hallucination_flagged == True).label("flagged_count"),
            func.avg(QueryLog.latency_ms).label("avg_latency_ms"),
        )
    )
    stats = stats_result.one()

    total = stats.total_queries or 1
    flag_rate = (stats.flagged_count or 0) / total
    avg_faith = float(stats.avg_faithfulness or 0)

    # ── Health status ─────────────────────────────────────────────────────────
    active_anomalies_result = await db.execute(
        select(func.count(PipelineAnomaly.id)).where(
            PipelineAnomaly.remediation_status == "pending"
        )
    )
    active_anomalies = active_anomalies_result.scalar() or 0

    if active_anomalies > 3 or flag_rate > 0.4 or avg_faith < 0.5:
        health_status = "critical"
    elif active_anomalies > 1 or flag_rate > 0.2 or avg_faith < 0.7:
        health_status = "degraded"
    else:
        health_status = "healthy"

    # ── Recent query score timeseries (last 20) ───────────────────────────────
    recent_queries_result = await db.execute(
        select(
            QueryLog.query_id,
            QueryLog.query_text,
            QueryLog.faithfulness_score,
            QueryLog.relevance_score,
            QueryLog.hallucination_score,
            QueryLog.is_hallucination_flagged,
            QueryLog.latency_ms,
            QueryLog.created_at,
        )
        .order_by(desc(QueryLog.created_at))
        .limit(20)
    )
    recent_queries = [
        {
            "query_id": r.query_id,
            "query_text": r.query_text[:60] + "..." if r.query_text and len(r.query_text) > 60 else r.query_text,
            "faithfulness_score": round(float(r.faithfulness_score or 0), 3),
            "relevance_score": round(float(r.relevance_score or 0), 3),
            "hallucination_score": round(float(r.hallucination_score or 0), 3),
            "is_flagged": r.is_hallucination_flagged,
            "latency_ms": r.latency_ms,
            "timestamp": r.created_at.isoformat() if r.created_at else None,
        }
        for r in recent_queries_result.all()
    ]

    # ── Document stats ────────────────────────────────────────────────────────
    doc_stats_result = await db.execute(
        select(Document.status, func.count(Document.id).label("count"))
        .group_by(Document.status)
    )
    doc_stats = {row.status: int(row.count) for row in doc_stats_result.all()}

    # ── Recent anomalies (last 10) ────────────────────────────────────────────
    anomalies_result = await db.execute(
        select(PipelineAnomaly)
        .order_by(desc(PipelineAnomaly.detected_at))
        .limit(10)
    )
    recent_anomalies = [
        {
            "id": a.id,
            "anomaly_type": a.anomaly_type,
            "severity": a.severity,
            "description": a.description,
            "root_cause": a.root_cause,
            "remediation_action": a.remediation_action,
            "remediation_status": a.remediation_status,
            "detected_at": a.detected_at.isoformat(),
        }
        for a in anomalies_result.scalars().all()
    ]

    # ── Drift stats ───────────────────────────────────────────────────────────
    try:
        drift_results = await drift_watchdog.run_corpus_drift_check()
    except Exception as e:
        logger.warning(f"Drift check failed: {e}")
        drift_results = []

    return {
        "health": {
            "status": health_status,
            "active_anomalies": active_anomalies,
        },
        "queries": {
            "total": stats.total_queries or 0,
            "avg_faithfulness": round(avg_faith, 3),
            "avg_relevance": round(float(stats.avg_relevance or 0), 3),
            "avg_hallucination_score": round(float(stats.avg_hallucination or 0), 3),
            "hallucination_flag_rate": round(flag_rate, 3),
            "avg_latency_ms": round(float(stats.avg_latency_ms or 0)),
            "recent": list(reversed(recent_queries)),  # chronological order
        },
        "documents": doc_stats,
        "anomalies": recent_anomalies,
        "drift": drift_results,
    }