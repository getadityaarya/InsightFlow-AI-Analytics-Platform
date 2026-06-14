"""
Upload API — Phase 1+2+3+4
Handles file upload, validation, profiling, schema extraction, and knowledge base creation.
Also persists Session metadata to SQL for observability.
"""

import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.session import Session
from app.services.ingestion import (
    load_dataframe, profile_dataframe, save_session_data,
    file_fingerprint, IngestionError,
)
from app.services.schema_intelligence import (
    infer_schema_metadata, store_schema, generate_create_table_sql,
)
from app.services.rag_store import index_schema, ensure_index

router = APIRouter()
logger = logging.getLogger(__name__)


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    tables: list[dict]
    profile_summary: dict
    message: str


@router.post("/", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    sid = session_id or str(uuid.uuid4())
    session_dir = Path(settings.UPLOAD_DIR) / sid
    session_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = session_dir / f"upload_{file.filename}"

    try:
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_FILE_SIZE_MB} MB limit.")
        with open(tmp_path, "wb") as f:
            f.write(content)

        # Phase 1: Load and validate
        try:
            dataframes = load_dataframe(str(tmp_path), file.filename)
        except IngestionError as e:
            raise HTTPException(status_code=422, detail=str(e))

        tables = []
        all_profiles = []

        for table_name, df in dataframes.items():
            # Phase 2: Profile
            profile = profile_dataframe(df, table_name)
            all_profiles.append(profile)

            # Persist DataFrame as parquet
            save_session_data(df, sid, table_name)

            # Phase 3+4: Schema intelligence → MongoDB
            schema_doc = infer_schema_metadata(df, table_name, sid)
            await store_schema(schema_doc)
            await ensure_index()
            await index_schema(schema_doc)   # Phase 4: Elasticsearch RAG

            tables.append({
                "table_name": table_name,
                "rows": profile["rows"],
                "columns": profile["columns"],
                "quality_score": profile["quality_score"],
                "missing_pct": profile["missing_values_pct"],
                "duplicate_pct": profile["duplicate_rows_pct"],
                "column_types": {c["name"]: c["inferred_type"] for c in profile["column_profiles"]},
                "create_table_sql": generate_create_table_sql(schema_doc),
            })

        profile_summary = {
            "total_tables": len(tables),
            "total_rows": sum(t["rows"] for t in tables),
            "total_columns": sum(t["columns"] for t in tables),
            "avg_quality_score": round(sum(t["quality_score"] for t in tables) / max(len(tables), 1), 1),
            "fingerprint": file_fingerprint(str(tmp_path)),
        }

        # Persist Session to SQL (for observability + dedup)
        session_record = Session(
            id=sid,
            filename=file.filename,
            table_names=[t["table_name"] for t in tables],
            total_rows=profile_summary["total_rows"],
            total_columns=profile_summary["total_columns"],
            quality_score=profile_summary["avg_quality_score"],
            fingerprint=profile_summary["fingerprint"],
        )
        # Upsert — replace if session_id reused
        existing = await db.get(Session, sid)
        if existing:
            for k, v in session_record.__dict__.items():
                if not k.startswith("_"):
                    setattr(existing, k, v)
        else:
            db.add(session_record)
        await db.commit()

        logger.info(f"Upload OK: session={sid}, tables={len(tables)}, rows={profile_summary['total_rows']}")

        return UploadResponse(
            session_id=sid,
            filename=file.filename,
            tables=tables,
            profile_summary=profile_summary,
            message=f"Successfully loaded {len(tables)} table(s) with {profile_summary['total_rows']:,} rows.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
