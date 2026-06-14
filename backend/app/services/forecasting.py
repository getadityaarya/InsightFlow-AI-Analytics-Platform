"""
Phase 15 — Forecasting Engine
Uses Facebook Prophet for time-series forecasting.
Automatically detects the date and value columns from the dataset.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ForecastError(Exception):
    pass


def run_forecast(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    periods: int = 30,
    freq: str = "D",
) -> dict:
    """
    Run Prophet forecast on a time series column.

    Args:
        df: DataFrame with the time series
        date_col: Name of the date column
        value_col: Name of the numeric column to forecast
        periods: Number of future periods to predict
        freq: Frequency — 'D' daily, 'W' weekly, 'M' monthly, 'Q' quarterly

    Returns:
        {
            "historical": DataFrame,
            "forecast": DataFrame (Prophet output),
            "summary": str,
            "confidence_interval": str,
        }
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ForecastError(
            "Prophet is not installed. Run: pip install prophet"
        )

    # Prepare Prophet input (requires 'ds' and 'y' columns)
    ts_df = df[[date_col, value_col]].copy()
    ts_df.columns = ["ds", "y"]
    ts_df["ds"] = pd.to_datetime(ts_df["ds"], errors="coerce")
    ts_df = ts_df.dropna().sort_values("ds")

    if len(ts_df) < 10:
        raise ForecastError(
            f"Not enough data to forecast. Need at least 10 data points, got {len(ts_df)}."
        )

    # Fit model
    try:
        model = Prophet(
            yearly_seasonality="auto",
            weekly_seasonality="auto",
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.80,
        )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(ts_df)
    except Exception as e:
        raise ForecastError(f"Prophet initialization or fitting failed: {e}")

    # Generate future dates
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # Summary stats
    last_actual = float(ts_df["y"].iloc[-1])
    forecast_end = float(forecast["yhat"].iloc[-1])
    trend_pct = ((forecast_end - last_actual) / max(abs(last_actual), 1)) * 100

    direction = "increase" if trend_pct > 0 else "decrease"
    summary = (
        f"Forecast for next {periods} {freq}-periods: "
        f"Expected {direction} of {abs(trend_pct):.1f}% "
        f"(from {last_actual:,.1f} to {forecast_end:,.1f})."
    )

    # Confidence interval
    last_row = forecast.iloc[-1]
    ci_str = (
        f"80% confidence interval at period end: "
        f"{last_row['yhat_lower']:,.1f} – {last_row['yhat_upper']:,.1f}"
    )

    return {
        "historical": ts_df,
        "forecast": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        "summary": summary,
        "confidence_interval": ci_str,
        "trend_pct": round(trend_pct, 2),
        "direction": direction,
    }


def detect_forecast_columns(df: pd.DataFrame) -> Optional[tuple[str, str]]:
    """
    Auto-detect the best (date_col, value_col) pair for forecasting.
    Returns None if no suitable pair found.
    """
    from app.services.ingestion import _infer_column_type

    date_cols = []
    numeric_cols = []

    for col in df.columns:
        col_type = _infer_column_type(df[col])
        if col_type == "datetime":
            date_cols.append(col)
        elif col_type in ("numeric", "currency"):
            numeric_cols.append(col)

    if not date_cols or not numeric_cols:
        return None

    # Prefer first date col and first currency/numeric col
    return (date_cols[0], numeric_cols[0])


def is_forecast_question(question: str) -> bool:
    """Detect if a question is asking for a forecast."""
    keywords = [
        "forecast", "predict", "projection", "trend", "future",
        "next month", "next quarter", "next year", "next 6", "next 12",
        "will be", "expected", "anticipate", "estimate future",
    ]
    question_lower = question.lower()
    return any(kw in question_lower for kw in keywords)


def parse_forecast_periods(question: str) -> tuple[int, str]:
    """
    Parse forecast period and frequency from natural language.
    Returns (periods, freq).
    """
    import re

    q_lower = question.lower()

    # Check frequency
    if any(w in q_lower for w in ["month", "monthly"]):
        freq = "M"
    elif any(w in q_lower for w in ["week", "weekly"]):
        freq = "W"
    elif any(w in q_lower for w in ["quarter", "quarterly"]):
        freq = "Q"
    else:
        freq = "D"

    # Check number of periods
    match = re.search(r"next\s+(\d+)", q_lower)
    if match:
        periods = int(match.group(1))
    else:
        # Default by frequency
        defaults = {"D": 30, "W": 12, "M": 6, "Q": 4}
        periods = defaults.get(freq, 30)

    return (periods, freq)
