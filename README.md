<div align="center">

# ARIA
### Adaptive RAG Intelligence & Auditing System

**A production-grade, self-healing RAG pipeline for AI research papers**


</div>

---

## What is ARIA?

ARIA is an intelligent research assistant that automatically ingests AI/ML papers from arXiv, answers questions about them with full citation tracing, audits every response for hallucinations using NLI scoring, detects when the corpus drifts or goes stale, and automatically remediates quality issues — with a human-in-the-loop gate for high-risk actions.

Most RAG systems only build the happy path: ingest → retrieve → generate. ARIA builds the full production stack — including the monitoring, auditing, and self-healing layers that make RAG systems actually trustworthy.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION LAYER                             │
│  arXiv API → PDF Downloader → Parser → Chunker → BGE Embedder      │
│                                                    ↓                │
│                                              Qdrant Vector Store    │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          QUERY LAYER                                │
│  User Query → Dense Retrieval → Cross-Encoder Reranker → Groq LLM  │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         AUDITOR LAYER                               │
│  NLI Faithfulness Scorer → Hallucination Detector → Citation Tracer │
│                    ↓ (if flagged)                                   │
│              Groq Judge LLM → Verdict + Reason                      │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    SELF-HEALING MONITOR                             │
│  Anomaly Detector → Root Cause LLM → Remediation Engine            │
│  Drift Watchdog   → Staleness Detector → Human-in-Loop Gate        │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      OBSERVABILITY LAYER                            │
│         Live Dashboard · Metrics API · Audit Logs · Reports        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 🔍 Intelligent Document Ingestion
- Auto-fetches papers from arXiv API using curated paper IDs across 5 topic clusters (RAG, LLM foundations, Agents, Multimodal, Evaluation)
- Recursive chunking with semantic overlap for dense research text
- Local GPU-accelerated embedding using `BAAI/bge-large-en-v1.5`

### 🧠 Production RAG Pipeline
- Dense vector retrieval from Qdrant with cosine similarity
- Cross-encoder reranking using `ms-marco-MiniLM` for precision
- Answer generation via Llama 3.3 70B on Groq API
- Full citation tracing — every answer sentence mapped back to its source chunk

### ✅ Hallucination Auditing (Novel)
- **NLI-based faithfulness scoring**: DeBERTa-v3-small checks if each answer sentence is entailed by source chunks
- **Dual-layer detection**: fast local NLI + Groq LLM judge for flagged cases
- **Relevance scoring**: measures semantic alignment between query and retrieved chunks
- Every query logged with full audit trail in PostgreSQL

### 🚨 Self-Healing Pipeline Monitor
- Rolling window anomaly detection: score drops, hallucination spikes, latency anomalies, ingestion failures
- Groq-powered root cause analysis: natural language diagnosis of what went wrong and why
- Automated remediation engine: re-chunk failing docs, purge low-quality chunks, rebuild index
- **Human-in-the-loop gate**: high-risk actions (rebuild_index, purge) require explicit approval via API

### 📡 Embedding Drift Detection
- Monitors semantic distribution of the corpus over time
- Flags when newly ingested documents drift from the corpus centroid
- Staleness detector alerts when the research corpus hasn't been updated

### 📊 Live Observability Dashboard
- Real-time pipeline health (healthy / degraded / critical)
- Score timelines: faithfulness, relevance, hallucination over last 20 queries
- Anomaly timeline with root cause and remediation status
- Document index stats and query latency charts

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI + Uvicorn | Async REST API |
| Embeddings | BAAI/bge-large-en-v1.5 | Local GPU inference |
| LLM | Llama 3.3 70B via Groq | Answer generation + judging |
| Vector Store | Qdrant | Semantic search |
| Database | PostgreSQL + SQLAlchemy | Audit logging |
| Streaming | Redis | Event queue |
| NLI Scorer | DeBERTa-v3-small | Faithfulness scoring |
| Reranker | ms-marco-MiniLM | Cross-encoder reranking |
| Containerization | Docker + Docker Compose | Deployment |

---

## Research Corpus

ARIA automatically ingests foundational AI/ML papers across 5 clusters:

| Cluster | Papers |
|---|---|
| **RAG** | RAG (Lewis et al.), DPR, CRAG, GraphRAG, RAG Survey |
| **LLM Foundations** | Attention Is All You Need, GPT-3, LLaMA 2, Mistral 7B, InstructGPT |
| **Agents** | ReAct, Toolformer, HuggingGPT, AgentBench |
| **Multimodal** | CLIP, LLaVA, BLIP-2, Flamingo |
| **Evaluation** | TruthfulQA, HELM, FActScore, RAGAS, Hallucination Survey |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop
- NVIDIA GPU (recommended — runs on CPU too)
- Free [Groq API key](https://console.groq.com)

### 1. Clone and configure
```bash
git clone https://github.com/YOUR_USERNAME/ARIA.git
cd ARIA
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### 2. Start infrastructure
```bash
docker-compose up -d qdrant postgres redis
```

### 3. Install dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate

pip install -r requirements.txt

# NVIDIA GPU (RTX 30xx/40xx/50xx):
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 4. Run ARIA
```bash
python main.py
```

Visit **http://localhost:8000/docs** for the full API.

### 5. Ingest your first papers
```bash
curl -X POST http://localhost:8000/arxiv/fetch \
  -H "Content-Type: application/json" \
  -d '{"cluster": "rag", "max_per_query": 3}'
```

### 6. Ask a question
```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main components of a RAG system?", "top_k": 4}'
```

---

## Full Docker Deployment

```bash
# Build and start everything including the ARIA app
docker-compose up --build

# Background mode
docker-compose up -d --build

# View logs
docker-compose logs -f aria
```

---

## API Reference

### arXiv Ingestion
| Endpoint | Method | Description |
|---|---|---|
| `/arxiv/clusters` | GET | List topic clusters and queries |
| `/arxiv/preview` | GET | Preview papers without ingesting |
| `/arxiv/fetch` | POST | Fetch and ingest papers (background) |
| `/arxiv/staleness` | GET | Check corpus age and freshness |

### Document Ingestion
| Endpoint | Method | Description |
|---|---|---|
| `/ingest/upload` | POST | Upload PDF/DOCX/TXT manually |
| `/ingest/status/{doc_id}` | GET | Check ingestion status |
| `/ingest/{doc_id}` | DELETE | Remove a document |

### Query & Audit
| Endpoint | Method | Description |
|---|---|---|
| `/query/` | POST | Submit question → answer + full audit |
| `/query/audit/{query_id}` | GET | Retrieve audit for past query |

### Monitor
| Endpoint | Method | Description |
|---|---|---|
| `/monitor/health` | GET | Pipeline health status |
| `/monitor/anomalies` | GET | List detected anomalies |
| `/monitor/check` | POST | Trigger anomaly check now |
| `/monitor/anomalies/{id}/diagnose` | POST | Run Groq root cause analysis |
| `/monitor/anomalies/{id}/approve` | POST | Approve high-risk remediation |
| `/monitor/anomalies/{id}/resolve` | POST | Manually resolve anomaly |

### Dashboard
| Endpoint | Method | Description |
|---|---|---|
| `/dashboard/metrics` | GET | All metrics for live dashboard |

---

## Project Structure

```
ARIA/
├── arxiv/
│   ├── fetcher.py          # arXiv API client with curated paper IDs
│   ├── downloader.py       # Concurrent PDF downloader
│   ├── pipeline.py         # End-to-end ingest orchestrator
│   ├── staleness.py        # Corpus freshness monitor
│   └── api.py              # arXiv REST endpoints
├── ingestion/
│   ├── api.py              # Manual upload endpoints
│   ├── chunker.py          # Recursive + fixed chunking strategies
│   ├── embedder.py         # BGE-large GPU embedder
│   └── parser.py           # PDF / DOCX / TXT parser
├── retrieval/
│   ├── query_api.py        # Query endpoint + Groq LLM integration
│   ├── retriever.py        # Dense Qdrant retrieval
│   └── reranker.py         # Cross-encoder reranker
├── auditor/
│   ├── auditor.py          # Audit orchestrator
│   ├── nli_scorer.py       # DeBERTa NLI faithfulness scorer
│   ├── groq_judge.py       # LLM-as-judge escalation
│   └── citation_tracer.py  # Answer-to-chunk mapping
├── monitor/
│   ├── anomaly_detector.py # Stateless anomaly detectors
│   ├── monitor.py          # Background monitoring loop (60s)
│   ├── root_cause.py       # Groq root cause analyzer
│   ├── remediation.py      # Auto-fix engine
│   ├── drift.py            # Embedding drift watchdog
│   ├── api.py              # Monitor + remediation endpoints
│   └── dashboard_api.py    # Metrics aggregation for dashboard
├── storage/
│   ├── postgres_client.py  # ORM models + async sessions
│   └── qdrant_client.py    # Vector store interface
├── config.py               # Central pydantic settings
├── main.py                 # FastAPI app entrypoint
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Build Phases

| Phase | Description |
|---|---|
| **Phase 1** | Core RAG pipeline — ingest, embed, retrieve, rerank, generate |
| **Phase 2** | Auditor layer — NLI scoring, Groq judge, citation tracing |
| **Phase 3** | Anomaly detection — score drops, hallucination spikes, latency |
| **Phase 4** | Root cause LLM + remediation engine + human-in-loop gate |
| **Phase 5** | Embedding drift watchdog + live observability dashboard |
| **Phase 6** | Docker deployment + documentation |
| **Phase 7** | arXiv auto-ingestion pipeline with curated research corpus |

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key — required | — |
| `GROQ_MODEL` | Model name | `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL` | HuggingFace model | `BAAI/bge-large-en-v1.5` |
| `EMBEDDING_DEVICE` | `cuda` or `cpu` | `cuda` |
| `QDRANT_HOST` | Qdrant hostname | `localhost` |
| `POSTGRES_URL` | PostgreSQL DSN | — |
| `REDIS_URL` | Redis DSN | — |
| `FAITHFULNESS_THRESHOLD` | Alert below this | `0.75` |
| `HALLUCINATION_FLAG_THRESHOLD` | Flag above this | `0.40` |
| `DRIFT_ALERT_THRESHOLD` | Cosine distance threshold | `0.25` |
| `ANOMALY_ROLLING_WINDOW` | Queries in rolling window | `50` |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built as an AI engineering project demonstrating production-grade RAG observability and self-healing systems.
</div>