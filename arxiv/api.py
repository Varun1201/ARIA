import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
from loguru import logger

from arxiv.fetcher import arxiv_fetcher, TOPIC_CLUSTERS
from arxiv.pipeline import arxiv_pipeline
from arxiv.staleness import staleness_detector

router = APIRouter(prefix="/arxiv", tags=["arXiv"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class FetchRequest(BaseModel):
    cluster: Optional[str] = None       # specific cluster or None for all
    max_per_query: int = 2              # papers per search query
    days_recent: Optional[int] = None  # if set, fetch only recent papers


class FetchResponse(BaseModel):
    message: str
    clusters: list[str]
    estimated_papers: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/clusters")
async def list_clusters():
    """List available topic clusters and their search queries."""
    return {
        "clusters": {
            name: queries
            for name, queries in TOPIC_CLUSTERS.items()
        },
        "total_clusters": len(TOPIC_CLUSTERS),
    }


@router.get("/staleness")
async def check_staleness():
    """Check if the research corpus has gone stale."""
    result = await staleness_detector.check_corpus_staleness()
    stats = await staleness_detector.get_corpus_stats()
    return {
        "staleness": result,
        "corpus_stats": stats,
    }


@router.post("/fetch")
async def fetch_and_ingest(
    request: FetchRequest,
    background_tasks: BackgroundTasks,
):
    """
    Fetch papers from arXiv and ingest them into ARIA.
    Runs in the background — returns immediately with job info.
    """
    if request.cluster and request.cluster not in TOPIC_CLUSTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown cluster '{request.cluster}'. Available: {list(TOPIC_CLUSTERS.keys())}"
        )

    if request.days_recent:
        clusters = ["recent"]
        estimated = request.days_recent * 3
        background_tasks.add_task(
            _run_recent_ingestion,
            days=request.days_recent,
        )
    elif request.cluster:
        clusters = [request.cluster]
        queries = TOPIC_CLUSTERS[request.cluster]
        estimated = len(queries) * request.max_per_query
        background_tasks.add_task(
            _run_cluster_ingestion,
            cluster=request.cluster,
            max_per_query=request.max_per_query,
        )
    else:
        clusters = list(TOPIC_CLUSTERS.keys())
        estimated = sum(len(q) for q in TOPIC_CLUSTERS.values()) * request.max_per_query
        background_tasks.add_task(
            _run_all_ingestion,
            max_per_query=request.max_per_query,
        )

    return FetchResponse(
        message=f"Ingestion started in background. Poll /arxiv/status for progress.",
        clusters=clusters,
        estimated_papers=estimated,
    )


@router.get("/preview")
async def preview_papers(
    cluster: str = "rag",
    max_results: int = 5,
):
    """
    Preview papers from arXiv without ingesting them.
    Useful for checking what would be fetched.
    """
    if cluster not in TOPIC_CLUSTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown cluster. Available: {list(TOPIC_CLUSTERS.keys())}"
        )

    queries = TOPIC_CLUSTERS[cluster]
    papers = await arxiv_fetcher.fetch_by_query(
        query=queries[0],
        max_results=max_results,
        topic_cluster=cluster,
    )

    return {
        "cluster": cluster,
        "query": queries[0],
        "papers": [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "authors": p.authors[:3],
                "published": p.published.isoformat(),
                "abstract": p.abstract[:300] + "...",
                "pdf_url": p.pdf_url,
                "categories": p.categories,
            }
            for p in papers
        ],
    }


# ── Background tasks ──────────────────────────────────────────────────────────

async def _run_cluster_ingestion(cluster: str, max_per_query: int):
    try:
        results = await arxiv_pipeline.run_cluster(cluster, max_per_query)
        indexed = sum(1 for r in results if r["status"] == "indexed")
        logger.info(f"Background ingestion complete: {indexed} papers indexed from '{cluster}'")
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}")


async def _run_all_ingestion(max_per_query: int):
    try:
        results = await arxiv_pipeline.run_all_clusters(max_per_query)
        total = sum(
            sum(1 for r in cluster_results if r["status"] == "indexed")
            for cluster_results in results.values()
        )
        logger.info(f"Full ingestion complete: {total} papers indexed")
    except Exception as e:
        logger.error(f"Full ingestion failed: {e}")


async def _run_recent_ingestion(days: int):
    try:
        results = await arxiv_pipeline.run_recent(days=days)
        indexed = sum(1 for r in results if r["status"] == "indexed")
        logger.info(f"Recent ingestion complete: {indexed} papers indexed")
    except Exception as e:
        logger.error(f"Recent ingestion failed: {e}")