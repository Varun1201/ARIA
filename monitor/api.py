from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from loguru import logger

from storage.postgres_client import get_db, PipelineAnomaly, QueryLog, Document
from monitor.monitor import pipeline_monitor
from monitor.root_cause import root_cause_analyzer
from monitor.remediation import remediation_engine, HIGH_RISK_ACTIONS

router = APIRouter(prefix="/monitor", tags=["Monitor"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ApproveRequest(BaseModel):
    confirmed: bool = False


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def pipeline_health(db: AsyncSession = Depends(get_db)):
    """Get overall pipeline health metrics."""
    recent_queries = await db.execute(
        select(
            func.count(QueryLog.id).label("total"),
            func.avg(QueryLog.faithfulness_score).label("avg_faithfulness"),
            func.avg(QueryLog.relevance_score).label("avg_relevance"),
            func.avg(QueryLog.hallucination_score).label("avg_hallucination"),
            func.count(QueryLog.id).filter(QueryLog.is_hallucination_flagged == True).label("flagged_count"),
            func.avg(QueryLog.latency_ms).label("avg_latency_ms"),
        )
    )
    stats = recent_queries.one()

    doc_stats = await db.execute(
        select(Document.status, func.count(Document.id).label("count"))
        .group_by(Document.status)
    )
    doc_status_counts = {row.status: int(row.count) for row in doc_stats.all()}

    active_anomalies = await db.execute(
        select(func.count(PipelineAnomaly.id)).where(
            PipelineAnomaly.remediation_status == "pending"
        )
    )
    active_count = active_anomalies.scalar()

    avg_faith = float(stats.avg_faithfulness or 0)
    total = stats.total or 1
    flag_rate = (stats.flagged_count or 0) / total

    if active_count > 3 or flag_rate > 0.4 or avg_faith < 0.5:
        health_status = "critical"
    elif active_count > 1 or flag_rate > 0.2 or avg_faith < 0.7:
        health_status = "degraded"
    else:
        health_status = "healthy"

    return {
        "status": health_status,
        "queries": {
            "total": stats.total or 0,
            "avg_faithfulness": round(avg_faith, 3),
            "avg_relevance": round(float(stats.avg_relevance or 0), 3),
            "avg_hallucination_score": round(float(stats.avg_hallucination or 0), 3),
            "hallucination_flag_rate": round(flag_rate, 3),
            "avg_latency_ms": round(float(stats.avg_latency_ms or 0)),
        },
        "documents": doc_status_counts,
        "active_anomalies": active_count,
    }


# ── Anomaly Management ────────────────────────────────────────────────────────

@router.get("/anomalies")
async def list_anomalies(
    status: str = "pending",
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """List pipeline anomalies filtered by status."""
    query = select(PipelineAnomaly).order_by(desc(PipelineAnomaly.detected_at)).limit(limit)
    if status != "all":
        query = query.where(PipelineAnomaly.remediation_status == status)

    result = await db.execute(query)
    anomalies = result.scalars().all()

    return {
        "count": len(anomalies),
        "anomalies": [
            {
                "id": a.id,
                "anomaly_type": a.anomaly_type,
                "severity": a.severity,
                "description": a.description,
                "root_cause": a.root_cause,
                "remediation_action": a.remediation_action,
                "remediation_status": a.remediation_status,
                "detected_at": a.detected_at.isoformat(),
                "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
                "extra_metadata": a.extra_metadata,
            }
            for a in anomalies
        ],
    }


@router.post("/check")
async def trigger_check():
    """Manually trigger an anomaly check right now."""
    await pipeline_monitor._run_checks()
    anomalies = await pipeline_monitor.get_active_anomalies()
    return {
        "message": "Anomaly check complete",
        "active_anomalies": len(anomalies),
        "anomalies": anomalies,
    }


# ── Root Cause Analysis ───────────────────────────────────────────────────────

@router.post("/anomalies/{anomaly_id}/diagnose")
async def diagnose_anomaly(anomaly_id: int, db: AsyncSession = Depends(get_db)):
    """
    Run root cause analysis on an anomaly using Groq LLM.
    Automatically recommends and queues a remediation action.
    """
    # Fetch anomaly
    result = await db.execute(
        select(PipelineAnomaly).where(PipelineAnomaly.id == anomaly_id)
    )
    anomaly = result.scalar_one_or_none()
    if not anomaly:
        raise HTTPException(status_code=404, detail=f"Anomaly {anomaly_id} not found")

    # Gather context — fix Decimal serialization with explicit float cast
    metrics_result = await db.execute(
        select(
            func.avg(QueryLog.faithfulness_score).label("avg_faithfulness"),
            func.avg(QueryLog.relevance_score).label("avg_relevance"),
            func.avg(QueryLog.hallucination_score).label("avg_hallucination"),
            func.avg(QueryLog.latency_ms).label("avg_latency_ms"),
        )
    )
    row = metrics_result.one()
    metrics = {
        "avg_faithfulness": round(float(row.avg_faithfulness or 0), 3),
        "avg_relevance": round(float(row.avg_relevance or 0), 3),
        "avg_hallucination": round(float(row.avg_hallucination or 0), 3),
        "avg_latency_ms": round(float(row.avg_latency_ms or 0), 1),
    }

    recent_logs_result = await db.execute(
        select(
            QueryLog.query_text,
            QueryLog.faithfulness_score,
            QueryLog.relevance_score,
            QueryLog.hallucination_score,
            QueryLog.is_hallucination_flagged,
            QueryLog.latency_ms,
        )
        .order_by(desc(QueryLog.created_at))
        .limit(5)
    )
    query_logs = [
        {
            "query_text": r.query_text,
            "faithfulness_score": float(r.faithfulness_score or 0),
            "relevance_score": float(r.relevance_score or 0),
            "hallucination_score": float(r.hallucination_score or 0),
            "is_hallucination_flagged": r.is_hallucination_flagged,
            "latency_ms": r.latency_ms,
        }
        for r in recent_logs_result.all()
    ]

    doc_stats_result = await db.execute(
        select(Document.status, func.count(Document.id).label("count"))
        .group_by(Document.status)
    )
    doc_stats = {row.status: int(row.count) for row in doc_stats_result.all()}

    # Run root cause analysis
    diagnosis = await root_cause_analyzer.analyze(
        anomaly_type=anomaly.anomaly_type,
        severity=anomaly.severity,
        description=anomaly.description,
        metrics=metrics,
        query_logs=query_logs,
        doc_stats=doc_stats,
    )

    # Save diagnosis to anomaly record
    anomaly.root_cause = diagnosis.get("root_cause")
    anomaly.remediation_action = diagnosis.get("recommended_action")
    await db.commit()

    # Auto-execute low-risk actions, gate high-risk ones
    action = diagnosis.get("recommended_action", "alert_only")
    risk_level = diagnosis.get("risk_level", "high")

    remediation_result = await remediation_engine.execute(
        anomaly_id=anomaly_id,
        action=action,
        risk_level=risk_level,
        approved=action not in HIGH_RISK_ACTIONS,
    )

    return {
        "anomaly_id": anomaly_id,
        "diagnosis": diagnosis,
        "remediation": remediation_result,
    }


# ── Human-in-Loop Approval ────────────────────────────────────────────────────

@router.post("/anomalies/{anomaly_id}/approve")
async def approve_remediation(
    anomaly_id: int,
    request: ApproveRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Approve a high-risk remediation action that was awaiting human approval.
    Requires confirmed=true in the request body.
    """
    if not request.confirmed:
        raise HTTPException(
            status_code=400,
            detail="Must set confirmed=true to approve remediation"
        )

    result = await db.execute(
        select(PipelineAnomaly).where(PipelineAnomaly.id == anomaly_id)
    )
    anomaly = result.scalar_one_or_none()
    if not anomaly:
        raise HTTPException(status_code=404, detail=f"Anomaly {anomaly_id} not found")

    if anomaly.remediation_status != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Anomaly is in '{anomaly.remediation_status}' status, not pending approval"
        )

    action = anomaly.remediation_action
    if not action:
        raise HTTPException(status_code=400, detail="No remediation action set — run /diagnose first")

    logger.info(f"Human approved remediation '{action}' for anomaly {anomaly_id}")

    remediation_result = await remediation_engine.execute(
        anomaly_id=anomaly_id,
        action=action,
        risk_level="high",
        approved=True,
    )

    anomaly.remediation_status = "approved"
    anomaly.resolved_at = datetime.utcnow()
    await db.commit()

    return {
        "anomaly_id": anomaly_id,
        "action": action,
        "result": remediation_result,
    }


@router.post("/anomalies/{anomaly_id}/resolve")
async def resolve_anomaly(anomaly_id: int, db: AsyncSession = Depends(get_db)):
    """Manually mark an anomaly as resolved without remediation."""
    result = await db.execute(
        select(PipelineAnomaly).where(PipelineAnomaly.id == anomaly_id)
    )
    anomaly = result.scalar_one_or_none()
    if not anomaly:
        raise HTTPException(status_code=404, detail=f"Anomaly {anomaly_id} not found")

    anomaly.remediation_status = "executed"
    anomaly.resolved_at = datetime.utcnow()
    await db.commit()
    return {"message": f"Anomaly {anomaly_id} resolved"}