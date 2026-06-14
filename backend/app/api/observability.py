"""Observability API — query logs, session stats, tool call history."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.models.session import QueryLog, ToolCall, Session

router = APIRouter()


@router.get("/stats/{session_id}")
async def get_session_stats(session_id: str, db: AsyncSession = Depends(get_db)):
    """Aggregated performance stats for a session."""
    result = await db.execute(
        select(
            func.count(QueryLog.id).label("total_queries"),
            func.avg(QueryLog.total_time_ms).label("avg_latency_ms"),
            func.sum(QueryLog.row_count).label("total_rows_fetched"),
            func.sum(QueryLog.llm_calls).label("total_llm_calls"),
            func.sum(QueryLog.is_cached).label("cached_hits"),
        ).where(QueryLog.session_id == session_id)
    )
    row = result.one()

    class_result = await db.execute(
        select(QueryLog.classification, func.count(QueryLog.id))
        .where(QueryLog.session_id == session_id)
        .group_by(QueryLog.classification)
    )
    breakdown = {r[0] or "unknown": r[1] for r in class_result.all()}

    errors = await db.execute(
        select(func.count(QueryLog.id))
        .where(QueryLog.session_id == session_id, QueryLog.error.isnot(None))
    )

    return {
        "session_id": session_id,
        "total_queries": row.total_queries or 0,
        "avg_latency_ms": round(row.avg_latency_ms or 0, 1),
        "total_rows_fetched": row.total_rows_fetched or 0,
        "total_llm_calls": row.total_llm_calls or 0,
        "cached_hits": row.cached_hits or 0,
        "error_count": errors.scalar() or 0,
        "classification_breakdown": breakdown,
    }


@router.get("/logs/{session_id}")
async def get_query_logs(
    session_id: str,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """Recent query logs for a session."""
    result = await db.execute(
        select(QueryLog)
        .where(QueryLog.session_id == session_id)
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
    )
    logs = result.scalars().all()
    return {
        "session_id": session_id,
        "logs": [
            {
                "id": log.id,
                "question": log.question,
                "classification": log.classification,
                "sql_valid": bool(log.sql_valid),
                "row_count": log.row_count,
                "chart_type": log.chart_type,
                "is_forecast": bool(log.is_forecast),
                "is_cached": bool(log.is_cached),
                "total_time_ms": log.total_time_ms,
                "error": log.error,
                "created_at": str(log.created_at),
            }
            for log in logs
        ],
    }
