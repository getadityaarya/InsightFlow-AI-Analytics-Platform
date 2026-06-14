"""
Phases 5+6+7 — SQL Generation, Validation, and Safe Execution
- SQL generated via Gemini with schema context (never hallucinating columns)
- Validated with sqlglot before execution
- Only SELECT allowed; DDL/DML blocked
- Results cached in memory per session
"""

import pandas as pd
import sqlite3
import sqlglot
import sqlglot.errors
import re
import json
import hashlib
import time
import logging
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.services.ingestion import load_session_data

logger = logging.getLogger(__name__)

# Simple in-process LRU cache (replace with Redis in production)
_query_cache: dict[str, dict] = {}

BLOCKED_STATEMENTS = {
    "DELETE", "DROP", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "MERGE", "EXEC", "EXECUTE",
}


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — SQL Validator
# ─────────────────────────────────────────────────────────────────────────────

class SQLValidationError(Exception):
    pass


def validate_sql(sql: str) -> str:
    """
    Parse and validate SQL for safety.
    Returns cleaned SQL or raises SQLValidationError.
    """
    sql_clean = sql.strip().rstrip(";")

    # Check for blocked statement types
    first_token = sql_clean.split()[0].upper() if sql_clean else ""
    if first_token in BLOCKED_STATEMENTS:
        raise SQLValidationError(
            f"Statement type '{first_token}' is not allowed. Only SELECT is permitted."
        )

    # Regex-based safety check for mutation keywords anywhere in query
    blocked_pattern = r"\b(" + "|".join(BLOCKED_STATEMENTS) + r")\b"
    if re.search(blocked_pattern, sql_clean, re.IGNORECASE):
        raise SQLValidationError(
            "Query contains disallowed keywords (DELETE, DROP, UPDATE, etc.)."
        )

    # Parse with sqlglot to catch syntax errors
    try:
        parsed = sqlglot.parse_one(sql_clean, dialect="sqlite")
        if parsed is None:
            raise SQLValidationError("Could not parse SQL.")
    except sqlglot.errors.ParseError as e:
        raise SQLValidationError(f"SQL syntax error: {e}") from e

    # Must be a SELECT statement
    if not isinstance(parsed, sqlglot.exp.Select):
        raise SQLValidationError("Only SELECT statements are permitted.")

    return sql_clean


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — Query Execution
# ─────────────────────────────────────────────────────────────────────────────

def execute_sql_on_dataframes(
    sql: str,
    dataframes: dict[str, pd.DataFrame],
    session_id: str,
    use_cache: bool = True,
) -> dict:
    """
    Execute validated SQL against in-memory DataFrames using DuckDB.
    Falls back to SQLite if DuckDB unavailable.

    Returns:
        {
            "columns": [...],
            "data": [[...], ...],
            "row_count": int,
            "execution_time_ms": float,
            "cached": bool,
        }
    """
    # Cache key
    cache_key = hashlib.md5(f"{session_id}:{sql}".encode()).hexdigest()
    if use_cache and cache_key in _query_cache:
        cached = _query_cache[cache_key]
        cached["cached"] = True
        return cached

    t0 = time.perf_counter()

    try:
        result_df = _execute_with_duckdb(sql, dataframes)
    except Exception as e:
        logger.warning(f"DuckDB execution failed ({e}), trying SQLite…")
        result_df = _execute_with_sqlite(sql, dataframes)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Truncate large results
    if len(result_df) > settings.MAX_SQL_ROWS:
        result_df = result_df.head(settings.MAX_SQL_ROWS)
        logger.warning(f"Result truncated to {settings.MAX_SQL_ROWS} rows")

    # Serialise
    columns = list(result_df.columns)
    data = result_df.values.tolist()

    # Convert non-serialisable types
    def safe(v):
        if pd.isna(v) if not isinstance(v, (list, dict)) else False:
            return None
        try:
            json.dumps(v)
            return v
        except (TypeError, ValueError):
            return str(v)

    data = [[safe(cell) for cell in row] for row in data]

    result = {
        "columns": columns,
        "data": data,
        "row_count": len(data),
        "execution_time_ms": round(elapsed_ms, 2),
        "cached": False,
    }

    _query_cache[cache_key] = result
    return result


def _execute_with_duckdb(sql: str, dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect(database=":memory:")
    for name, df in dataframes.items():
        con.register(name, df)
    return con.execute(sql).df()


def _execute_with_sqlite(sql: str, dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fallback: load DataFrames into an in-memory SQLite and query."""
    conn = sqlite3.connect(":memory:")
    for name, df in dataframes.items():
        df.to_sql(name, conn, if_exists="replace", index=False)
    result = pd.read_sql_query(sql, conn)
    conn.close()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Session data loading helper
# ─────────────────────────────────────────────────────────────────────────────

def load_all_tables(session_id: str, table_names: list[str]) -> dict[str, pd.DataFrame]:
    """Load all tables for a session from parquet files."""
    frames = {}
    for tbl in table_names:
        try:
            frames[tbl] = load_session_data(session_id, tbl)
        except FileNotFoundError:
            logger.warning(f"Table not found: {tbl}")
    return frames


def result_to_dataframe(result: dict) -> pd.DataFrame:
    """Convert execution result dict back to DataFrame."""
    return pd.DataFrame(result["data"], columns=result["columns"])
