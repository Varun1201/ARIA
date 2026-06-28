import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config import settings
from storage.postgres_client import init_db
from storage.qdrant_client import qdrant
from ingestion.api import router as ingest_router
from retrieval.query_api import router as query_router
from monitor.api import router as monitor_router
from monitor.dashboard_api import router as dashboard_router
from monitor.monitor import pipeline_monitor
from arxiv.api import router as arxiv_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ARIA...")
    await init_db()
    logger.info("PostgreSQL tables ready")
    await qdrant.ensure_collection()
    logger.info("Qdrant collection ready")
    monitor_task = asyncio.create_task(pipeline_monitor.start())
    logger.info("Pipeline monitor started")
    logger.info("ARIA is ready")
    yield
    pipeline_monitor.stop()
    monitor_task.cancel()
    logger.info("Shutting down ARIA...")


app = FastAPI(
    title="ARIA — Adaptive RAG Intelligence & Auditing System",
    description="A self-healing RAG pipeline for AI research papers with hallucination auditing, anomaly detection, and automated remediation.",
    version="0.7.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(monitor_router)
app.include_router(dashboard_router)
app.include_router(arxiv_router)


@app.get("/health", tags=["System"])
async def health():
    qdrant_info = await qdrant.get_collection_info()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": "0.7.0",
        "embedding_model": settings.embedding_model,
        "llm": settings.groq_model,
        "qdrant": qdrant_info,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)