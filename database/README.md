# Database Architecture

InsightFlow AI uses three storage systems, each chosen for its strengths.

## SQLite / PostgreSQL (Relational)
**Purpose:** Session metadata, query logs, tool call observability  
**File:** `insightflow.db` (SQLite) or configure `DATABASE_URL` for PostgreSQL  
**Tables:**

| Table | Purpose |
|---|---|
| `sessions` | Tracks upload sessions, filenames, row/col counts, quality scores |
| `query_logs` | Every query with latency, SQL, row count, cache hits, errors |
| `tool_calls` | Individual MCP tool invocations per query |

**Run migrations:**
```bash
python database/migrate.py
# or via Makefile:
make migrate
```

## MongoDB (Document Store)
**Purpose:** Schema metadata (column types, RAG text) + conversation memory  
**Collections:**

| Collection | Purpose |
|---|---|
| `schemas` | Per-table schema docs with column metadata and RAG text |
| `memory` | Conversation history: question, SQL, insight, timestamp |
| `sessions` | Session-level metadata cache |

**Start locally:**
```bash
docker run -d -p 27017:27017 --name insightflow-mongo mongo:7.0
```

## Elasticsearch (Vector/Full-text Search)
**Purpose:** Phase 4 RAG — retrieve relevant schema chunks for each query  
**Index:** `insightflow_schema`  
**Fallback:** In-memory keyword search (no Elasticsearch required for development)

**Start locally:**
```bash
docker run -d -p 9200:9200 -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.13.0
```

## Data Flow

```
Upload CSV
    │
    ▼
Parquet (disk)          ← Raw DataFrame stored per session
    │
    ├─► MongoDB schemas ← Column types, descriptions, RAG text
    │
    ├─► Elasticsearch   ← RAG text embedded for semantic search
    │
    └─► SQLite sessions ← Row count, quality score, fingerprint

Ask Question
    │
    ▼
Elasticsearch search    ← Retrieve relevant schema chunks (RAG)
    │
    ▼
MongoDB memory          ← Past 5 conversation turns (context)
    │
    ▼
Gemini SQL generation   ← Schema + history + question → SQL
    │
    ▼
DuckDB execution        ← Fast in-memory columnar query
    │
    ▼
SQLite query_logs       ← Latency, rows, chart type logged
    │
    ▼
MongoDB memory          ← Store this turn for future context
```
