# InsightFlow AI — Deployment Guide

## Local Development (Quickstart)

### Prerequisites
- Python 3.11+
- MongoDB 7.0 (via Docker or local install)
- Elasticsearch 8.x *(optional — RAG falls back gracefully without it)*

### 1. Start infrastructure

```bash
# MongoDB only (minimal)
docker run -d -p 27017:27017 --name mongo mongo:7.0

# MongoDB + Elasticsearch (full RAG support)
docker compose up mongo elasticsearch -d
```

### 2. Configure environment

```bash
cd backend
cp .env.example .env
```

Edit `.env` — minimum required:

```env
GEMINI_API_KEY=your-key-here      # https://aistudio.google.com/app/apikey
MONGODB_URL=mongodb://localhost:27017
```

Optional but recommended:

```env
TAVILY_API_KEY=your-key-here      # https://tavily.com — enables WEB/HYBRID queries
ELASTIC_URL=http://localhost:9200  # enables semantic RAG search
```

### 3. Install dependencies

```bash
pip install -r requirements.txt

# Optional: real semantic embeddings (vs hash fallback)
pip install sentence-transformers
```

### 4. Start backend

```bash
uvicorn app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 5. Start frontend

```bash
cd ../frontend
python -m http.server 3000
# Open: http://localhost:3000
```

Or use the Makefile:

```bash
make dev        # backend
make frontend   # frontend (separate terminal)
```

---

## Docker Compose (Production-like)

```bash
# Copy and configure
cp backend/.env.example .env
# Add GEMINI_API_KEY and TAVILY_API_KEY

# Start all services
docker compose up --build

# With dev UIs (Kibana + Mongo Express)
docker compose --profile dev up --build
```

**Services:**

| Service | URL | Notes |
|---------|-----|-------|
| Frontend | http://localhost:80 | Nginx-served SPA |
| Backend API | http://localhost:8000 | FastAPI |
| API Docs | http://localhost:8000/docs | Swagger UI |
| MongoDB | localhost:27017 | Schema + memory storage |
| Elasticsearch | http://localhost:9200 | RAG vector index |
| Kibana (dev) | http://localhost:5601 | ES UI |
| Mongo Express (dev) | http://localhost:8081 | MongoDB UI (admin/admin123) |

---

## Production Deployment

### Environment variables for production

```env
DEBUG=false
SECRET_KEY=<64-char random string>
DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/insightflow
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/
ELASTIC_URL=https://your-deployment.es.us-east-1.aws.found.io
ELASTIC_API_KEY=your-elastic-cloud-api-key
CORS_ORIGINS=["https://yourdomain.com"]
```

### Recommended production stack

```
Internet → CloudFlare (CDN + WAF)
              ↓
         Nginx / ALB
         /           \
   Frontend        Backend (2+ workers)
   (S3/CDN)        uvicorn --workers 4
                      |          |          |
                  MongoDB    Elasticsearch  PostgreSQL
                  Atlas      Elastic Cloud  RDS
```

### Scaling workers

```bash
# Production: 4 workers, bound to 0.0.0.0
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Behind gunicorn (more robust process management)
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

> **Note:** With multiple workers, replace the in-process query cache in
> `sql_engine.py` with Redis: `pip install redis[hiredis]` and use
> `aioredis` for async cache operations.

### Redis cache (multi-worker)

```python
# In sql_engine.py, replace _query_cache dict with:
import aioredis, json

async def get_cache(key: str):
    r = aioredis.from_url(settings.REDIS_URL)
    val = await r.get(key)
    return json.loads(val) if val else None

async def set_cache(key: str, value: dict, ttl: int = 3600):
    r = aioredis.from_url(settings.REDIS_URL)
    await r.setex(key, ttl, json.dumps(value))
```

---

## Running Tests

```bash
cd backend

# Fast tests (no Prophet, no external services)
pytest tests/ -v -m "not slow"

# All tests
pytest tests/ -v

# Coverage report
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html

# Specific phase tests
pytest tests/test_insightflow.py -v          # Core phases 1-8, 15
pytest tests/test_extended.py -v             # Phases 11-17
```

---

## Generating Sample Data

```bash
python scripts/generate_sample_data.py
```

Creates in `sample_data/`:

| File | Rows | Description |
|------|------|-------------|
| `ecommerce_sales.csv` | 2,000 | Sales with seasonality + trends |
| `hr_employees.csv` | 800 | HR/attrition dataset |
| `marketing_campaigns.csv` | 3,000 | Campaign performance |
| `sample_store.sqlite` | 3 tables | Multi-table orders/customers/products |

---

## Troubleshooting

### "Session not found" after restart
Session parquet files are stored in `./uploads/{session_id}/`. If the
backend restarts without a volume mount, files are lost. Either:
- Mount `./uploads` as a persistent volume (already done in `docker-compose.yml`)
- Re-upload the dataset

### Elasticsearch unavailable
InsightFlow falls back to full-schema retrieval automatically. You'll see
in the logs:
```
Semantic search failed — falling back to full schema
```
No action needed. To enable RAG: start Elasticsearch and set `ELASTIC_URL`.

### Gemini API errors
- Check your `GEMINI_API_KEY` is valid and has quota
- The model `gemini-1.5-pro` requires a paid Google AI Studio plan
- For free tier: change `GEMINI_MODEL=gemini-1.5-flash` in `.env`

### Prophet installation fails
Prophet requires a C++ compiler. On Ubuntu:
```bash
apt-get install -y gcc g++ libstdc++6
pip install prophet
```
On Mac: `xcode-select --install` first.

### Large file uploads timeout
Increase Nginx `proxy_read_timeout` in `frontend/nginx.conf` (already set to 300s).
For very large files (>100MB), increase `MAX_FILE_SIZE_MB` in `.env`.
