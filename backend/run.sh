#!/usr/bin/env bash
# InsightFlow AI — Backend startup script
# Usage: ./run.sh [dev|prod]

set -e

MODE=${1:-dev}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════╗"
echo "║       InsightFlow AI — Backend           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check .env file
if [ ! -f .env ]; then
  echo "⚠️  No .env file found — copying from .env.example"
  cp .env.example .env
  echo "   → Edit .env and add your GEMINI_API_KEY, then re-run."
  echo ""
fi

# Check GEMINI_API_KEY
source .env 2>/dev/null || true
if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your-gemini-api-key-here" ]; then
  echo "⚠️  GEMINI_API_KEY not set in .env"
  echo "   Get your key at: https://aistudio.google.com/app/apikey"
  echo "   The server will start in MOCK mode (no real LLM calls)."
  echo ""
fi

# Check MongoDB
echo "🔍 Checking MongoDB..."
if command -v mongosh &>/dev/null; then
  if mongosh --quiet --eval "db.adminCommand('ping')" &>/dev/null; then
    echo "✅ MongoDB is running"
  else
    echo "⚠️  MongoDB not running — starting via Docker..."
    docker run -d -p 27017:27017 --name insightflow-mongo mongo:7.0 2>/dev/null || \
    docker start insightflow-mongo 2>/dev/null || \
    echo "   Could not start MongoDB. Install or run: docker run -d -p 27017:27017 mongo:7.0"
  fi
else
  echo "⚠️  mongosh not found — ensure MongoDB is running on port 27017"
fi

echo ""

# Run DB migrations
echo "📦 Running database migrations..."
python database/migrate.py 2>/dev/null || python -c "
import asyncio, sys
sys.path.insert(0, '.')
from app.core.database import engine, Base, init_db
asyncio.run(init_db())
print('✅ Database initialised')
"

echo ""

if [ "$MODE" = "prod" ]; then
  echo "🚀 Starting in PRODUCTION mode (2 workers)..."
  exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --loop uvloop \
    --no-access-log
else
  echo "🔧 Starting in DEVELOPMENT mode (hot reload)..."
  echo "   API docs: http://localhost:8000/docs"
  echo "   Health:   http://localhost:8000/api/health"
  echo ""
  exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir app \
    --log-level info
fi
