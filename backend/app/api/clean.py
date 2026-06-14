"""
Clean & Export API
- GET  /api/clean/export/{session_id}/{table_name}  → download as CSV or Excel
- POST /api/clean/{session_id}/{table_name}          → clean data in-place
- GET  /api/clean/{session_id}/{table_name}/outliers → detect outliers

NOTE: /export route is defined FIRST so FastAPI does not treat "export"
      as a session_id value when matching path parameters.
"""

import io
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.ingestion import load_session_data, save_session_data
from app.services.cleaning import clean_dataframe, detect_outliers, coerce_column_types

router = APIRouter()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Export endpoint — MUST be defined before /{session_id} path-param routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/export/{session_id}/{table_name}")
async def export_table(
    session_id: str,
    table_name: str,
    format: str = Query("csv", pattern="^(csv|excel)$"),
    max_rows: int = Query(100_000, ge=1, le=1_000_000),
):
    """
    Export a full table as CSV or Excel (xlsx).
    Streams the file directly — no temp file on disk.
    Usage: GET /api/clean/export/{session_id}/{table_name}?format=csv
    """
    try:
        df = load_session_data(session_id, table_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Table not found.")

    df = df.head(max_rows)
    safe_name = table_name.replace(" ", "_")

    if format == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}.csv"'},
        )

    else:  # excel
        import pandas as pd
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=table_name[:31])
        buf.seek(0)
        return StreamingResponse(
            iter([buf.read()]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}.xlsx"'},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Clean endpoint
# ─────────────────────────────────────────────────────────────────────────────

class CleanRequest(BaseModel):
    strategy: str = "auto"          # "auto" | "conservative" | "aggressive"
    custom_rules: Optional[dict] = None  # {"col": "median"|"mode"|"drop"|"zero"|"unknown"}
    coerce_types: bool = True


@router.post("/{session_id}/{table_name}")
async def clean_table(
    session_id: str,
    table_name: str,
    request: CleanRequest,
):
    """
    Clean a table in-place: impute missing values, remove duplicates,
    trim whitespace, coerce types.  Overwrites the session parquet.
    """
    try:
        df = load_session_data(session_id, table_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Table not found.")

    cleaned_df, report = clean_dataframe(
        df,
        strategy=request.strategy,
        custom_rules=request.custom_rules,
    )

    if request.coerce_types:
        cleaned_df, coerce_changes = coerce_column_types(cleaned_df)
        if coerce_changes:
            report["operations"].extend(coerce_changes)

    save_session_data(cleaned_df, session_id, table_name)

    return {
        "session_id": session_id,
        "table_name": table_name,
        "cleaning_report": report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Outlier detection endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{session_id}/{table_name}/outliers")
async def get_outliers(
    session_id: str,
    table_name: str,
    method: str = Query("iqr", pattern="^(iqr|zscore)$"),
):
    """Detect outliers in all numeric columns using IQR or Z-score."""
    try:
        df = load_session_data(session_id, table_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Table not found.")

    results = detect_outliers(df, method=method)
    return {
        "session_id": session_id,
        "table_name": table_name,
        "method": method,
        "outliers": results,
        "total_affected_columns": len(results),
    }
