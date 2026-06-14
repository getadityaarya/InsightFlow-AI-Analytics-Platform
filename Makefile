# InsightFlow AI — Developer Makefile
# Usage: make <target>

.PHONY: help install dev test lint docker-up docker-down clean seed

PYTHON := python3
PIP    := pip3
UVICORN := uvicorn

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────────

install:  ## Install Python dependencies
	cd backend && $(PIP) install -r requirements.txt

install-dev:  ## Install + dev extras
	cd backend && $(PIP) install -r requirements.txt pytest pytest-asyncio httpx

# ── Run ──────────────────────────────────────────────────────────────────────

dev:  ## Start backend in dev mode (hot reload)
	cd backend && $(UVICORN) app.main:app --reload --host 0.0.0.0 --port 8000

frontend:  ## Serve frontend (Python simple server)
	cd frontend && $(PYTHON) -m http.server 3000

# ── Test ─────────────────────────────────────────────────────────────────────

test:  ## Run all tests
	cd backend && $(PYTHON) -m pytest tests/ -v

test-fast:  ## Run tests excluding slow (Prophet) tests
	cd backend && $(PYTHON) -m pytest tests/ -v -m "not slow"

test-coverage:  ## Run tests with coverage report
	cd backend && $(PYTHON) -m pytest tests/ --cov=app --cov-report=html -v

# ── Code Quality ──────────────────────────────────────────────────────────────

lint:  ## Lint with ruff
	cd backend && ruff check app/ tests/

format:  ## Format with black
	cd backend && black app/ tests/

# ── Docker ───────────────────────────────────────────────────────────────────

docker-up:  ## Start all services with Docker Compose
	docker compose up --build -d

docker-up-dev:  ## Start with MongoDB Express dev UI
	docker compose --profile dev up --build -d

docker-down:  ## Stop all services
	docker compose down

docker-logs:  ## Tail backend logs
	docker compose logs -f backend

docker-restart:  ## Restart backend only
	docker compose restart backend

# ── Seed Data ─────────────────────────────────────────────────────────────────

migrate:  ## Run database migrations
	cd backend && python database/migrate.py

seed:  ## Generate sample CSV datasets for testing
	$(PYTHON) scripts/generate_sample_data.py

# ── Clean ────────────────────────────────────────────────────────────────────

clean:  ## Remove generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf backend/.pytest_cache backend/htmlcov backend/.coverage
	rm -rf backend/insightflow.db
	rm -rf backend/uploads/*
