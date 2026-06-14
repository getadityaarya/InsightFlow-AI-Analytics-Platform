"""
Phase 12 — Conversation Memory Service
Stores and retrieves question/SQL/result history so the agent can
answer follow-up questions like "compare with previous quarter".
"""

import time
import logging
from typing import Optional
from sqlalchemy import select, func

from app.core.database import AsyncSessionLocal
from app.models.session import Memory

logger = logging.getLogger(__name__)


async def store_memory(
    session_id: str,
    question: str,
    classification: str,
    sql: Optional[str],
    result_preview: str,
    insight: str,
    chart_type: Optional[str] = None,
) -> str:
    """Persist a Q&A turn to SQLite."""
    async with AsyncSessionLocal() as db:
        mem = Memory(
            session_id=session_id,
            question=question,
            classification=classification,
            sql=sql,
            result_preview=result_preview,
            insight=insight,
            chart_type=chart_type,
        )
        db.add(mem)
        await db.commit()
        await db.refresh(mem)
        return str(mem.id)


async def get_conversation_history(
    session_id: str,
    limit: int = 10,
) -> list[dict]:
    """Retrieve recent conversation turns, newest last."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Memory)
            .filter(Memory.session_id == session_id)
            .order_by(Memory.timestamp.asc())
            .limit(limit)
        )
        records = result.scalars().all()
        return [
            {
                "session_id": r.session_id,
                "timestamp": r.timestamp.timestamp(),
                "question": r.question,
                "classification": r.classification,
                "sql": r.sql,
                "result_preview": r.result_preview,
                "insight": r.insight,
                "chart_type": r.chart_type,
            }
            for r in records
        ]


async def get_last_sql(session_id: str) -> Optional[str]:
    """Get the most recent SQL query executed in this session."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Memory.sql)
            .filter(Memory.session_id == session_id)
            .filter(Memory.sql.is_not(None))
            .order_by(Memory.timestamp.desc())
            .limit(1)
        )
        sql = result.scalar_one_or_none()
        return sql


async def search_memory(session_id: str, keyword: str) -> list[dict]:
    """
    Full-text search across conversation history for a keyword.
    Useful for referencing past queries (e.g. "what was the revenue last time?").
    """
    async with AsyncSessionLocal() as db:
        # SQLite LIKE is case-insensitive by default for ASCII
        result = await db.execute(
            select(Memory)
            .filter(Memory.session_id == session_id)
            .filter(Memory.question.ilike(f"%{keyword}%"))
            .order_by(Memory.timestamp.desc())
            .limit(5)
        )
        records = result.scalars().all()
        return [
            {
                "session_id": r.session_id,
                "timestamp": r.timestamp.timestamp(),
                "question": r.question,
                "classification": r.classification,
                "sql": r.sql,
                "result_preview": r.result_preview,
                "insight": r.insight,
                "chart_type": r.chart_type,
            }
            for r in records
        ]


async def clear_session_memory(session_id: str) -> int:
    """Clear all memory for a session."""
    from sqlalchemy import delete
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            delete(Memory).where(Memory.session_id == session_id)
        )
        await db.commit()
        return result.rowcount


async def get_session_summary(session_id: str) -> dict:
    """Return a quick summary of session activity."""
    async with AsyncSessionLocal() as db:
        total = await db.scalar(
            select(func.count(Memory.id)).where(Memory.session_id == session_id)
        )
        db_queries = await db.scalar(
            select(func.count(Memory.id))
            .where(Memory.session_id == session_id)
            .where(Memory.classification == "DATABASE")
        )
        web_queries = await db.scalar(
            select(func.count(Memory.id))
            .where(Memory.session_id == session_id)
            .where(Memory.classification.in_(["WEB", "HYBRID"]))
        )
        return {
            "total_questions": total or 0,
            "database_queries": db_queries or 0,
            "web_queries": web_queries or 0,
        }
