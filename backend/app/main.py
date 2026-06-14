"""
InsightFlow AI — Agentic Data Intelligence Platform
Main FastAPI Application
"""

from collections import defaultdict
from contextlib import asynccontextmanager
import time
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os

from app.api import upload, query, profile, health, observability, clean, sessions, mcp
from app.core.config import settings
from app.core.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting InsightFlow AI...")
    await init_db()
    yield
    logger.info("Shutting down InsightFlow AI...")


app = FastAPI(
    title="InsightFlow AI",
    description="Agentic Data Intelligence Platform — Chat with your business data",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-process rate limiter ────────────────────────────────────────────────────
_rate_store: dict = defaultdict(list)
RATE_LIMIT_QUERY  = 30   # requests / minute / IP
RATE_LIMIT_UPLOAD = 10


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if settings.DEBUG:               # disabled in dev / test
        return await call_next(request)
    path = request.url.path
    if path.startswith("/api/query") or path.startswith("/api/upload"):
        ip  = request.client.host if request.client else "unknown"
        key = f"{ip}:{path.split('/')[2]}"
        now = time.time()
        _rate_store[key] = [t for t in _rate_store[key] if now - t < 60]
        limit = RATE_LIMIT_QUERY if "query" in path else RATE_LIMIT_UPLOAD
        if len(_rate_store[key]) >= limit:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Max {limit} requests/minute."},
            )
        _rate_store[key].append(now)
    return await call_next(request)


# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health.router,        prefix="/api",            tags=["Health"])
app.include_router(upload.router,        prefix="/api/upload",     tags=["Upload"])
app.include_router(query.router,         prefix="/api/query",      tags=["Query"])
app.include_router(profile.router,       prefix="/api/profile",    tags=["Profile"])
app.include_router(clean.router,         prefix="/api/clean",      tags=["Clean & Export"])
app.include_router(sessions.router,      prefix="/api/sessions",   tags=["Sessions"])
app.include_router(mcp.router,           prefix="/api/mcp",        tags=["MCP Tools"])
app.include_router(observability.router, prefix="/api/observe",    tags=["Observability"])


@app.get("/")
async def root():
    return {
        "name": "InsightFlow AI",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "health":       "/api/health",
            "upload":       "/api/upload/",
            "query":        "/api/query/",
            "profile":      "/api/profile/{session_id}/{table_name}",
            "clean":        "/api/clean/{session_id}/{table_name}",
            "export":       "/api/clean/export/{session_id}/{table_name}",
            "sessions":     "/api/sessions/",
            "mcp_tools":    "/api/mcp/tools",
            "mcp_invoke":   "/api/mcp/invoke",
            "observe":      "/api/observe/stats/{session_id}",
        },
    }

# ── Static Files (Frontend) ────────────────────────────────────────────────────
frontend_path = os.path.join(os.path.dirname(__file__), "../../frontend")
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
