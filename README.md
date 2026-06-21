# ⚡ InsightFlow AI — Agentic Data Intelligence Platform

> Upload any business dataset. Ask questions in plain English. Get SQL, charts, forecasts, and business insights — powered by Gemini.

---

## 🏗️ Core Architecture & Design Decisions

This project is built with a focus on performance, security, and scalability. Here are the key architectural choices:

### ✅ Implemented Features
| Phase | Decision | Status |
|---|---|---|
| Data Ingestion (CSV/Excel/SQLite) | Comprehensive data source support | ✅ Implemented |
| Data Profiling | Critical differentiator | ✅ Full quality scoring |
| Schema Intelligence | Business meaning mapping | ✅ SQLite storage |
| RAG Knowledge Base | Prevents SQL hallucination | ✅ FAISS semantic search |
| SQL Generation + Validation | Security-first approach | ✅ sqlglot + allowlist |
| Chart Engine | Auto-selection logic | ✅ 5 chart types |
| Business Insight Engine | Goes beyond standard analytics | ✅ Gemini narration |
| Question Classifier | Intelligent routing | ✅ DATABASE/WEB/HYBRID |
| Memory System | Contextual follow-ups | ✅ SQLite persistence |
| Agent Planner | Truly agentic workflow | ✅ LangGraph orchestration |
| Clarification Agent | Handles ambiguity | ✅ JSON-structured options |
| Forecasting Engine | Prophet time-series | ✅ Auto period detection |
| MCP Tools | Extensible tool design | ✅ 6 tools defined |
| Observability | Comprehensive logging | ✅ SQLAlchemy logs |

### 🔧 Architectural Highlights

1. **DuckDB SQL Engine** — Uses DuckDB (in-memory columnar) with SQLite fallback for query execution, ensuring fast processing on large datasets compared to standard pandas processing.

2. **Parquet Persistence** — Session data is persisted to Parquet files keyed by session ID, allowing the system to survive server restarts efficiently.

3. **Cache Layer** — Query results are cached by SHA-256 hash of (session_id + SQL). Subsequent identical queries return instantly.

4. **Prophet Auto-Detection** — Automatically detects the optimal (date_col, value_col) pair for time-series forecasting without requiring manual user input.

5. **Multi-Layer SQL Security** — Employs regex-based keyword blocking as a primary layer before sqlglot AST parsing to prevent destructive operations.

6. **Automated Quality Scoring** — Quantifies data quality (0–100) based on missing values, duplicates, and dataset size, providing immediate feedback in the UI upon upload.

7. **Async-First Design** — All DB calls, LLM requests, and external searches are fully asynchronous, ensuring high concurrency without blocking the event loop.

8. **LangGraph Orchestration** — Uses LangGraph to manage complex state transitions and conditional routing for the agentic reasoning loop.

---

## 🚀 Quick Start

### Option A — Local (Recommended for Development)

```bash
# 1. Clone and navigate
git clone https://github.com/yourname/insightflow-ai
cd insightflow-ai

# 2. Start MongoDB (required)
docker run -d -p 27017:27017 --name mongo mongo:7.0

# 3. Install backend
cd backend
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env: add GEMINI_API_KEY (required)

# 5. Start backend
uvicorn app.main:app --reload --port 8000

# 6. Open frontend (in a new terminal)
cd ../frontend
python -m http.server 3000
# Open: http://localhost:3000
```

### Option B — Docker Compose (Full Stack)

```bash
# Copy and configure environment
cp backend/.env.example .env
# Add your GEMINI_API_KEY to .env

# Start everything
docker compose up --build

# Open: http://localhost:80
# API docs: http://localhost:8000/docs
# MongoDB UI: docker compose --profile dev up (port 8081)
```

---

## 🗂️ Project Structure

```
insightflow-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                    ← FastAPI app + CORS + lifespan
│   │   ├── core/
│   │   │   ├── config.py              ← All env-driven settings
│   │   │   └── database.py            ← SQLAlchemy async + MongoDB motor
│   │   ├── api/
│   │   │   ├── upload.py              ← POST /api/upload/
│   │   │   ├── query.py               ← POST /api/query/
│   │   │   ├── profile.py             ← GET  /api/profile/{session}/{table}
│   │   │   └── health.py              ← GET  /api/health
│   │   ├── agents/
│   │   │   └── orchestrator.py        ← Phase 13: full agent planner
│   │   ├── services/
│   │   │   ├── ingestion.py           ← Phase 1+2: load, validate, profile
│   │   │   ├── schema_intelligence.py ← Phase 3+4: metadata + RAG text
│   │   │   ├── sql_engine.py          ← Phase 5+6+7: generate, validate, execute
│   │   │   ├── llm_service.py         ← All Gemini calls (SQL, insight, classify)
│   │   │   ├── chart_engine.py        ← Phase 8: auto chart selection + Plotly
│   │   │   ├── memory.py              ← Phase 12: MongoDB conversation memory
│   │   │   ├── forecasting.py         ← Phase 15: Prophet time-series
│   │   │   ├── web_search.py          ← Phase 11: Tavily web search
│   │   │   └── observability.py       ← Phase 17: query logs + Dynatrace
│   │   ├── tools/
│   │   │   └── mcp_tools.py           ← Phase 16: MCP tool definitions + executor
│   │   ├── models/
│   │   │   └── session.py             ← SQLAlchemy: Session, QueryLog, ToolCall
│   │   └── prompts/
│   │       └── templates.py           ← All LLM prompt templates
│   ├── tests/
│   │   └── test_insightflow.py        ← 40+ tests across all phases
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── pytest.ini
│   └── .env.example
├── frontend/
│   ├── index.html                     ← Full SPA (vanilla JS + Plotly)
│   ├── Dockerfile
│   └── nginx.conf
├── scripts/
│   └── generate_sample_data.py        ← Generate test CSVs
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## 🔄 Full Request Flow

```
User: "Why did revenue decline in Q3?"
        │
        ▼
[1] Upload check → session_id exists?
        │
        ▼
[2] Schema context → MongoDB fetch → RAG text
        │
        ▼
[3] Classify → HYBRID (needs data + external context)
        │
        ▼
[4] Ambiguity check → is "revenue" clear? → Yes, proceed
        │
        ▼
[5] Plan:
    ① Query database for Q3 revenue trend
    ② Search web for Q3 economic context
    ③ Synthesise findings
        │
        ▼
[6a] Generate SQL → Gemini (with schema context)
[6b] Validate SQL → sqlglot (only SELECT allowed)
[6c] Execute → DuckDB (in-memory, 10ms)
        │
        ▼
[7] Auto-select chart → Line chart (time series detected)
[8] Build Plotly config → sent to frontend
        │
        ▼
[9] Web search → Tavily → top 5 results
        │
        ▼
[10] Synthesise → Gemini (internal + external findings)
        │
        ▼
[11] Store memory → MongoDB (question, SQL, insight)
        │
        ▼
[12] Log observability → SQLite QueryLog
        │
        ▼
Response → SQL + Chart + Insight + Web sources + Plan trace
```

---

## 🧪 Running Tests

```bash
cd backend

# All tests (fast only — no Prophet)
pytest tests/ -v -m "not slow"

# Full suite including forecasting
pytest tests/ -v

# Coverage report
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

**Test coverage:**
- ✅ Phase 1: Ingestion (CSV, Excel, SQLite, validation)
- ✅ Phase 2: Profiling (missing values, duplicates, quality score)
- ✅ Phase 3: Column type inference (6 types)
- ✅ Phase 3+4: Schema metadata generation
- ✅ Phase 6: SQL validation (10 security tests)
- ✅ Phase 7: SQL execution (DuckDB + SQLite fallback)
- ✅ Phase 8: Chart auto-selection (5 chart types)
- ✅ Phase 15: Forecasting (question detection, period parsing, column detection)
- ✅ Integration: end-to-end CSV → profile → schema → SQL pipelines

---

## 🔑 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/upload/` | Upload CSV/Excel/SQLite (Phases 1–4) |
| `POST` | `/api/query/` | Natural language query (Phases 5–15) |
| `GET` | `/api/query/history/{session_id}` | Conversation history (Phase 12) |
| `DELETE` | `/api/query/history/{session_id}` | Clear history |
| `GET` | `/api/query/schema/{session_id}` | Schema metadata |
| `GET` | `/api/profile/{session_id}/{table}` | Data quality profile (Phase 2) |
| `POST` | `/api/clean/{session_id}/{table}` | Clean data (impute, deduplicate) |
| `GET` | `/api/clean/{session_id}/{table}/outliers` | Detect outliers |
| `GET` | `/api/clean/export/{session_id}/{table}` | Export as CSV or Excel |
| `GET` | `/api/sessions/` | List all sessions |
| `GET` | `/api/sessions/{session_id}` | Get session metadata |
| `DELETE` | `/api/sessions/{session_id}` | Delete session (all data) |
| `GET` | `/api/mcp/tools` | List MCP tool definitions (Phase 16) |
| `POST` | `/api/mcp/invoke` | Invoke an MCP tool |
| `GET` | `/api/mcp/health` | RAG/Elasticsearch health |
| `GET` | `/api/observe/stats/{session_id}` | Observability stats (Phase 17) |
| `GET` | `/api/observe/logs/{session_id}` | Query logs |

Interactive API docs: `http://localhost:8000/docs`

---

## ⚙️ Configuration Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | — | Google Gemini API key |
| `TAVILY_API_KEY` | Optional | — | Web search (enables WEB/HYBRID) |
| `MONGODB_URL` | ✅ Yes | `localhost:27017` | Schema + memory storage |
| `DATABASE_URL` | Optional | SQLite | For query logs + sessions |
| `MAX_FILE_SIZE_MB` | Optional | `100` | Upload file size limit |
| `DYNATRACE_URL` | Optional | — | Observability export |
| `GEMINI_MODEL` | Optional | `gemini-1.5-pro` | Model selection |

---

## 🛣️ Production Roadmap

| Priority | Item | Effort |
|----------|------|--------|
| High | Replace in-process query cache with Redis | 2h |
| High | Add rate limiting (slowapi) | 1h |
| High | JWT authentication + user sessions | 4h |
| Medium | Elasticsearch embedding for true vector search | 1d |
| Medium | LangGraph migration for parallel tool execution | 2d |
| Medium | PostgreSQL migration from SQLite | 2h |
| Low | Dynatrace dashboard setup | 4h |
| Low | Export query results to CSV/Excel | 3h |
| Low | Slack/Teams bot integration | 1d |

---

## 🏆 Hackathon Differentiators

1. **Data Profiling** — Most competitors skip this. Data quality is scored immediately on upload.
2. **Security-first SQL** — Two-layer validation (regex + sqlglot AST). No DDL ever executes.
3. **Truly agentic** — Question → Classify → Plan → Tools → Synthesise. Not just prompt → answer.
4. **Hybrid evidence** — Internal data and external web search kept separate in the response.
5. **Forecasting** — Prophet integration with automatic column detection and period parsing.
6. **Observability** — Every query logged with latency, tool calls, and error tracking.
