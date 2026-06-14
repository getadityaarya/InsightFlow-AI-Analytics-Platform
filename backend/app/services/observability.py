"""
Phase 17 — Observability Service
Tracks every query, tool call, latency, and error.
Stores to SQL (QueryLog model) and optionally to Dynatrace.
"""

import time
import logging
import asyncio
from typing import Optional, Any
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.models.session import QueryLog, ToolCall

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Query observability
# ─────────────────────────────────────────────────────────────────────────────

async def log_query(
    db: AsyncSession,
    session_id: str,
    question: str,
    classification: Optional[str] = None,
    sql_generated: Optional[str] = None,
    sql_valid: bool = True,
    row_count: int = 0,
    chart_type: Optional[str] = None,
    is_forecast: bool = False,
    is_cached: bool = False,
    total_time_ms: float = 0.0,
    sql_time_ms: float = 0.0,
    llm_calls: int = 0,
    error: Optional[str] = None,
) -> int:
    """Persist a query log record. Returns the log ID."""
    log = QueryLog(
        session_id=session_id,
        question=question[:1000],
        classification=classification,
        sql_generated=sql_generated,
        sql_valid=1 if sql_valid else 0,
        row_count=row_count,
        chart_type=chart_type,
        is_forecast=1 if is_forecast else 0,
        is_cached=1 if is_cached else 0,
        total_time_ms=total_time_ms,
        sql_time_ms=sql_time_ms,
        llm_calls=llm_calls,
        error=error,
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)

    # Fire-and-forget Dynatrace export
    if settings.DYNATRACE_URL and settings.DYNATRACE_TOKEN:
        asyncio.create_task(_export_to_dynatrace(log))

    return log.id


async def log_tool_call(
    db: AsyncSession,
    query_log_id: int,
    tool_name: str,
    input_summary: str = "",
    latency_ms: float = 0.0,
    success: bool = True,
    error: Optional[str] = None,
):
    """Log an individual MCP tool invocation."""
    tc = ToolCall(
        query_log_id=query_log_id,
        tool_name=tool_name,
        input_summary=input_summary[:500],
        latency_ms=latency_ms,
        success=1 if success else 0,
        error=error,
    )
    db.add(tc)
    await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Timing context manager
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Simple context manager for timing code blocks."""
    def __init__(self):
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Analytics queries
# ─────────────────────────────────────────────────────────────────────────────

async def get_session_stats(db: AsyncSession, session_id: str) -> dict:
    """Return aggregated statistics for a session."""
    from sqlalchemy import select, func

    result = await db.execute(
        select(
            func.count(QueryLog.id).label("total_queries"),
            func.avg(QueryLog.total_time_ms).label("avg_latency_ms"),
            func.sum(QueryLog.row_count).label("total_rows_fetched"),
            func.sum(QueryLog.llm_calls).label("total_llm_calls"),
        ).where(QueryLog.session_id == session_id)
    )
    row = result.one()

    # Classification breakdown
    class_result = await db.execute(
        select(QueryLog.classification, func.count(QueryLog.id))
        .where(QueryLog.session_id == session_id)
        .group_by(QueryLog.classification)
    )
    classification_breakdown = {r[0]: r[1] for r in class_result.all()}

    return {
        "total_queries": row.total_queries or 0,
        "avg_latency_ms": round(row.avg_latency_ms or 0, 1),
        "total_rows_fetched": row.total_rows_fetched or 0,
        "total_llm_calls": row.total_llm_calls or 0,
        "classification_breakdown": classification_breakdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dynatrace export (optional)
# ─────────────────────────────────────────────────────────────────────────────

async def _export_to_dynatrace(log: QueryLog):
    """
    Push metrics to Dynatrace via their Events API.
    Only called when DYNATRACE_URL and DYNATRACE_TOKEN are configured.
    """
    try:
        import httpx
        payload = {
            "eventType": "CUSTOM_INFO",
            "title": "InsightFlow Query",
            "properties": {
                "session_id": log.session_id,
                "classification": log.classification,
                "total_time_ms": log.total_time_ms,
                "row_count": log.row_count,
                "sql_valid": log.sql_valid,
                "is_forecast": log.is_forecast,
                "error": log.error or "",
            },
        }
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{settings.DYNATRACE_URL}/api/v2/events/ingest",
                json=payload,
                headers={"Authorization": f"Api-Token {settings.DYNATRACE_TOKEN}"},
                timeout=5.0,
            )
    except Exception as e:
        logger.warning(f"Dynatrace export failed (non-critical): {e}")
