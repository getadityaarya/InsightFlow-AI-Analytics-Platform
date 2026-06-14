"""
Phase 1+2 — Data Ingestion & Profiling Service
Handles CSV, Excel, SQLite uploads with validation and quality reporting.
"""

import pandas as pd
import sqlite3
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Tuple
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


class IngestionError(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1  — Load & validate
# ─────────────────────────────────────────────────────────────────────────────

def load_dataframe(file_path: str, filename: str) -> dict[str, pd.DataFrame]:
    """
    Returns a dict of {table_name: DataFrame}.
    CSV/Excel = single table "main"; SQLite = one entry per table.
    """
    ext = Path(filename).suffix.lower().lstrip(".")

    if ext not in settings.ALLOWED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type: .{ext}")

    size = os.path.getsize(file_path)
    if size > MAX_BYTES:
        raise IngestionError(
            f"File too large ({size / 1e6:.1f} MB). Limit is {settings.MAX_FILE_SIZE_MB} MB."
        )

    try:
        if ext == "csv":
            df = pd.read_csv(file_path, low_memory=False)
            _validate_df(df)
            return {"main": df}

        elif ext in ("xlsx", "xls"):
            sheets = pd.read_excel(file_path, sheet_name=None)
            for name, df in sheets.items():
                _validate_df(df)
            return sheets

        elif ext in ("sqlite", "db"):
            conn = sqlite3.connect(file_path)
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            if tables.empty:
                raise IngestionError("SQLite file contains no tables.")
            result = {}
            for tbl in tables["name"]:
                df = pd.read_sql(f"SELECT * FROM [{tbl}]", conn)
                _validate_df(df)
                result[tbl] = df
            conn.close()
            return result

    except IngestionError:
        raise
    except Exception as e:
        raise IngestionError(f"Failed to read file: {e}") from e


def _validate_df(df: pd.DataFrame):
    if df is None or df.empty:
        raise IngestionError("File contains no data.")
    if len(df.columns) == 0:
        raise IngestionError("File has no columns.")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2  — Data Profiling (most hackathon projects skip this — we don't)
# ─────────────────────────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame, table_name: str = "main") -> dict:
    """
    Generates a comprehensive quality report for a DataFrame.
    """
    rows, cols = df.shape
    total_cells = rows * cols

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing.sum() / max(total_cells, 1)) * 100

    # Duplicates
    dup_rows = int(df.duplicated().sum())
    dup_pct = round(dup_rows / max(rows, 1) * 100, 2)

    # Column-level analysis
    col_profiles = []
    for col in df.columns:
        series = df[col]
        col_type = _infer_column_type(series)
        profile = {
            "name": col,
            "dtype": str(series.dtype),
            "inferred_type": col_type,
            "missing_count": int(missing[col]),
            "missing_pct": round(missing[col] / max(rows, 1) * 100, 2),
            "unique_count": int(series.nunique()),
            "unique_pct": round(series.nunique() / max(rows, 1) * 100, 2),
        }

        if col_type in ("numeric", "currency"):
            profile.update({
                "mean": _safe_round(series.mean()),
                "median": _safe_round(series.median()),
                "std": _safe_round(series.std()),
                "min": _safe_round(series.min()),
                "max": _safe_round(series.max()),
            })
        elif col_type == "datetime":
            profile.update({
                "min_date": str(pd.to_datetime(series, errors="coerce").min()),
                "max_date": str(pd.to_datetime(series, errors="coerce").max()),
            })
        elif col_type in ("text", "category"):
            top = series.value_counts().head(5)
            profile["top_values"] = top.index.tolist()

        col_profiles.append(profile)

    # Quality score (0–100)
    quality_score = _compute_quality_score(
        missing_pct=missing_pct,
        dup_pct=dup_pct,
        rows=rows,
    )

    return {
        "table_name": table_name,
        "rows": rows,
        "columns": cols,
        "total_cells": total_cells,
        "missing_values_count": int(missing.sum()),
        "missing_values_pct": round(missing_pct, 2),
        "duplicate_rows": dup_rows,
        "duplicate_rows_pct": dup_pct,
        "quality_score": quality_score,
        "column_profiles": col_profiles,
    }


def _infer_column_type(series: pd.Series) -> str:
    """Infer business-meaningful column type."""
    name_lower = series.name.lower() if series.name else ""
    dtype_str = str(series.dtype)

    # Datetime signals
    date_keywords = ["date", "time", "created", "updated", "timestamp", "at", "on"]
    if any(k in name_lower for k in date_keywords):
        if pd.to_datetime(series, errors="coerce").notna().mean() > 0.8:
            return "datetime"

    # Currency / revenue signals (whole-word match to avoid "cost" matching "count")
    import re as _re
    currency_keywords = ["amount", "revenue", "price", "cost", "sales", "profit",
                          "income", "spend", "fee", "total", "value", "gmv"]
    if any(_re.search(r'\b' + k + r'\b', name_lower) for k in currency_keywords) and ("float" in dtype_str or "int" in dtype_str):
        return "currency"

    # Numeric
    if "float" in dtype_str or "int" in dtype_str:
        return "numeric"

    # Try parsing as datetime
    if "object" in dtype_str:
        try:
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() > 0.7:
                return "datetime"
        except Exception:
            pass

    # Category vs text
    if series.nunique() / max(len(series), 1) < 0.05:
        return "category"

    return "text"


def _safe_round(val, digits: int = 4):
    try:
        return round(float(val), digits)
    except Exception:
        return None


def _compute_quality_score(missing_pct: float, dup_pct: float, rows: int) -> int:
    score = 100.0
    score -= min(missing_pct * 2, 40)      # penalise missing values
    score -= min(dup_pct * 1.5, 20)        # penalise duplicates
    if rows < 10:
        score -= 20                         # very small dataset penalty
    return max(0, int(score))


# ─────────────────────────────────────────────────────────────────────────────
# Session helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_session_data(df: pd.DataFrame, session_id: str, table_name: str) -> str:
    """Persist DataFrame to a parquet file for the session."""
    path = Path(settings.UPLOAD_DIR) / session_id
    path.mkdir(parents=True, exist_ok=True)
    dest = path / f"{table_name}.parquet"
    df.to_parquet(dest, index=False)
    return str(dest)


def load_session_data(session_id: str, table_name: str) -> pd.DataFrame:
    """Load DataFrame from session parquet."""
    path = Path(settings.UPLOAD_DIR) / session_id / f"{table_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Session data not found: {session_id}/{table_name}")
    return pd.read_parquet(path)


def file_fingerprint(file_path: str) -> str:
    """SHA-256 fingerprint for dedup detection."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
