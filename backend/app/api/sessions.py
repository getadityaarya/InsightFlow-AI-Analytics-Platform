"""
Sessions API
List, inspect, and delete upload sessions.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from pathlib import Path
import shutil

from app.core.database import get_db
from app.core.config import settings
from app.models.session import Session, QueryLog, Memory, SchemaMetadata
from app.services.vector_store import delete_session_index

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
async def list_sessions(db: AsyncSession = Depends(get_db)):
    """List all upload sessions."""
    result = await db.execute(
        select(Session).order_by(Session.created_at.desc())
    )
    sessions = result.scalars().all()
    return {
        "sessions": [
            {
                "id": s.id,
                "filename": s.filename,
                "table_names": s.table_names,
                "total_rows": s.total_rows,
                "total_columns": s.total_columns,
                "quality_score": s.quality_score,
                "created_at": str(s.created_at),
            }
            for s in sessions
        ],
        "count": len(sessions),
    }


@router.get("/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get metadata for a specific session."""
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {
        "id": session.id,
        "filename": session.filename,
        "table_names": session.table_names,
        "total_rows": session.total_rows,
        "total_columns": session.total_columns,
        "quality_score": session.quality_score,
        "fingerprint": session.fingerprint,
        "created_at": str(session.created_at),
        "last_active": str(session.last_active),
    }


@router.delete("/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """
    Fully delete a session:
    - SQL records (Session, QueryLog, Memory, SchemaMetadata)
    - Parquet files on disk
    - Elasticsearch/FAISS index entries
    """
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    deleted = {"sql": False, "disk": False, "vector_store": False}

    # 1. Delete SQL records
    try:
        await db.execute(delete(QueryLog).where(QueryLog.session_id == session_id))
        await db.execute(delete(Memory).where(Memory.session_id == session_id))
        await db.execute(delete(SchemaMetadata).where(SchemaMetadata.session_id == session_id))
        await db.delete(session)
        await db.commit()
        deleted["sql"] = True
    except Exception as e:
        logger.error(f"SQL delete failed: {e}")

    # 2. Delete disk files (parquet + any uploads)
    try:
        session_dir = Path(settings.UPLOAD_DIR) / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
        deleted["disk"] = True
    except Exception as e:
        logger.warning(f"Disk delete failed: {e}")

    # 4. Delete Elasticsearch entries
    try:
        deleted["elasticsearch"] = await delete_session_index(session_id)
    except Exception as e:
        logger.warning(f"Elasticsearch delete failed: {e}")

    logger.info(f"Session deleted: {session_id}")
    return {"session_id": session_id, "deleted": deleted}
