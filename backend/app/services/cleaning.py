"""
Data Cleaning Service
Provides automated and user-directed data cleaning operations.
Covers: missing value imputation, duplicate removal, type coercion, outlier flagging.
Called from the /api/clean endpoint and surfaced in the chat as a suggested action
when the profiler detects data quality issues.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Main cleaning entry point
# ─────────────────────────────────────────────────────────────────────────────

def clean_dataframe(
    df: pd.DataFrame,
    strategy: str = "auto",
    custom_rules: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply cleaning operations to a DataFrame.

    Args:
        df: Input DataFrame
        strategy: "auto" | "conservative" | "aggressive"
            - auto: drop fully empty cols, median-fill numerics, mode-fill categoricals
            - conservative: only drop fully empty rows/cols, no imputation
            - aggressive: drop all missing, deduplicate, coerce types
        custom_rules: dict of per-column rules, e.g.:
            {"sales_amount": "median", "status": "mode", "notes": "drop"}

    Returns:
        (cleaned_df, cleaning_report)
    """
    report = {
        "original_rows": len(df),
        "original_cols": len(df.columns),
        "operations": [],
        "rows_removed": 0,
        "cols_removed": 0,
        "values_imputed": 0,
    }

    cleaned = df.copy()

    # 1. Drop fully empty columns
    empty_cols = [c for c in cleaned.columns if cleaned[c].isna().all()]
    if empty_cols:
        cleaned = cleaned.drop(columns=empty_cols)
        report["operations"].append(f"Dropped {len(empty_cols)} fully empty column(s): {empty_cols}")
        report["cols_removed"] += len(empty_cols)

    # 2. Drop fully empty rows
    empty_rows_before = len(cleaned)
    cleaned = cleaned.dropna(how="all")
    empty_rows_dropped = empty_rows_before - len(cleaned)
    if empty_rows_dropped:
        report["operations"].append(f"Dropped {empty_rows_dropped} fully empty row(s)")
        report["rows_removed"] += empty_rows_dropped

    # 3. Deduplicate
    if strategy in ("auto", "aggressive"):
        dup_count = cleaned.duplicated().sum()
        if dup_count:
            cleaned = cleaned.drop_duplicates()
            report["operations"].append(f"Removed {dup_count} duplicate row(s)")
            report["rows_removed"] += dup_count

    # 4. Apply column-level imputation
    imputed_total = 0
    for col in cleaned.columns:
        rule = (custom_rules or {}).get(col)
        missing = cleaned[col].isna().sum()
        if missing == 0:
            continue

        if rule == "drop":
            cleaned = cleaned.dropna(subset=[col])
            report["operations"].append(f"Dropped {missing} rows with missing '{col}'")
            report["rows_removed"] += missing
            continue

        if strategy == "conservative":
            continue  # no imputation in conservative mode

        # Auto-determine imputation strategy
        from app.services.ingestion import _infer_column_type
        col_type = _infer_column_type(cleaned[col])

        if rule == "median" or (rule is None and col_type in ("numeric", "currency")):
            fill_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(fill_val)
            report["operations"].append(
                f"Imputed {missing} missing '{col}' values with median ({fill_val:.2f})"
            )
            imputed_total += missing

        elif rule == "mean" or (rule is None and col_type in ("numeric", "currency") and strategy == "aggressive"):
            fill_val = cleaned[col].mean()
            cleaned[col] = cleaned[col].fillna(fill_val)
            report["operations"].append(
                f"Imputed {missing} missing '{col}' values with mean ({fill_val:.2f})"
            )
            imputed_total += missing

        elif rule == "mode" or (rule is None and col_type in ("category", "text")):
            mode_vals = cleaned[col].mode()
            if not mode_vals.empty:
                fill_val = mode_vals.iloc[0]
                cleaned[col] = cleaned[col].fillna(fill_val)
                report["operations"].append(
                    f"Imputed {missing} missing '{col}' values with mode ('{fill_val}')"
                )
                imputed_total += missing

        elif rule == "zero":
            cleaned[col] = cleaned[col].fillna(0)
            report["operations"].append(f"Imputed {missing} missing '{col}' with 0")
            imputed_total += missing

        elif rule == "unknown":
            cleaned[col] = cleaned[col].fillna("Unknown")
            report["operations"].append(f"Imputed {missing} missing '{col}' with 'Unknown'")
            imputed_total += missing

        elif col_type == "datetime":
            # Forward-fill dates (most sensible for time series)
            cleaned[col] = cleaned[col].fillna(method="ffill")
            imputed_after = cleaned[col].isna().sum()
            filled = missing - imputed_after
            if filled:
                report["operations"].append(f"Forward-filled {filled} missing '{col}' dates")
                imputed_total += filled

    report["values_imputed"] = imputed_total

    # 5. Trim whitespace from string columns
    str_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        stripped = cleaned[col].str.strip() if hasattr(cleaned[col], "str") else cleaned[col]
        if not stripped.equals(cleaned[col]):
            cleaned[col] = stripped
            report["operations"].append(f"Trimmed whitespace in '{col}'")

    # 6. Summary
    report["final_rows"] = len(cleaned)
    report["final_cols"] = len(cleaned.columns)
    report["missing_remaining"] = int(cleaned.isna().sum().sum())

    if not report["operations"]:
        report["operations"].append("No cleaning operations needed — data is already clean.")

    return cleaned, report


# ─────────────────────────────────────────────────────────────────────────────
# Outlier detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_outliers(df: pd.DataFrame, method: str = "iqr") -> dict:
    """
    Detect outliers in numeric columns using IQR or Z-score method.

    Returns:
        {
            "column_name": {
                "outlier_count": int,
                "outlier_pct": float,
                "lower_bound": float,
                "upper_bound": float,
                "example_values": [...]
            }
        }
    """
    from app.services.ingestion import _infer_column_type

    results = {}
    for col in df.columns:
        col_type = _infer_column_type(df[col])
        if col_type not in ("numeric", "currency"):
            continue
        series = df[col].dropna()
        if len(series) < 10:
            continue

        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:  # z-score
            mean, std = series.mean(), series.std()
            lower, upper = mean - 3 * std, mean + 3 * std

        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) == 0:
            continue

        results[col] = {
            "outlier_count": len(outliers),
            "outlier_pct": round(len(outliers) / len(series) * 100, 2),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "example_values": [round(float(v), 4) for v in outliers.head(5).tolist()],
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Type coercion helpers
# ─────────────────────────────────────────────────────────────────────────────

def coerce_column_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Attempt to coerce columns to their most appropriate dtype.
    Returns (coerced_df, list_of_changes).
    """
    cleaned = df.copy()
    changes = []

    for col in cleaned.columns:
        original_dtype = str(cleaned[col].dtype)

        # Try numeric coercion for object columns
        if cleaned[col].dtype == object:
            numeric_attempt = pd.to_numeric(cleaned[col], errors="coerce")
            if numeric_attempt.notna().mean() > 0.9:
                cleaned[col] = numeric_attempt
                changes.append(f"'{col}': object → numeric")
                continue

            # Try datetime coercion
            try:
                dt_attempt = pd.to_datetime(cleaned[col], errors="coerce", infer_datetime_format=True)
                if dt_attempt.notna().mean() > 0.9:
                    cleaned[col] = dt_attempt
                    changes.append(f"'{col}': object → datetime")
            except Exception:
                pass

    return cleaned, changes
