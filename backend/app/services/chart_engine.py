"""
Phase 8 — Chart Engine
Automatically selects the best chart type and generates Plotly config
based on data shape and column types.
"""

import pandas as pd
import json
import logging
from typing import Optional

from app.services.ingestion import _infer_column_type

logger = logging.getLogger(__name__)

# Chart type decision rules (priority order)
CHART_RULES = [
    # Time series → Line chart
    {
        "name": "line",
        "condition": lambda cols, types: any(t == "datetime" for t in types) and len(cols) >= 2,
        "label": "Line Chart",
        "icon": "📈",
    },
    # Exactly two cols (1 category + 1 numeric) → Pie chart
    {
        "name": "pie",
        "condition": lambda cols, types: len(cols) == 2
                                          and sum(t in ("text", "category") for t in types) == 1
                                          and sum(t in ("numeric", "currency") for t in types) == 1,
        "label": "Pie Chart",
        "icon": "🥧",
    },
    # One+ text cols + one numeric → Bar chart
    {
        "name": "bar",
        "condition": lambda cols, types: sum(t in ("text", "category") for t in types) >= 1
                                         and sum(t in ("numeric", "currency") for t in types) >= 1
                                         and len(cols) <= 4,
        "label": "Bar Chart",
        "icon": "📊",
    },
    # Two numerics → Scatter plot
    {
        "name": "scatter",
        "condition": lambda cols, types: sum(t in ("numeric", "currency") for t in types) >= 2,
        "label": "Scatter Plot",
        "icon": "⚡",
    },
    # Fallback → Table
    {
        "name": "table",
        "condition": lambda cols, types: True,
        "label": "Data Table",
        "icon": "📋",
    },
]


def select_chart_type(df: pd.DataFrame) -> str:
    """Auto-select the best chart type for a result DataFrame."""
    if df is None or df.empty:
        return "table"

    cols = list(df.columns)
    types = [_infer_column_type(df[c]) for c in cols]

    for rule in CHART_RULES:
        if rule["condition"](cols, types):
            logger.info(f"Auto-selected chart: {rule['name']} for columns {cols}")
            return rule["name"]

    return "table"


def build_plotly_config(
    df: pd.DataFrame,
    chart_type: str,
    title: str = "",
) -> dict:
    """
    Generate a Plotly.js-compatible config dict for the frontend to render.
    The frontend renders this with Plotly.newPlot(div, data, layout).
    """
    if df is None or df.empty:
        return _empty_chart(title)

    cols = list(df.columns)
    types = [_infer_column_type(df[c]) for c in cols]

    # Identify axes
    datetime_cols = [c for c, t in zip(cols, types) if t == "datetime"]
    numeric_cols = [c for c, t in zip(cols, types) if t in ("numeric", "currency")]
    category_cols = [c for c, t in zip(cols, types) if t in ("text", "category")]

    # Limit chart data points
    chart_df = df.head(500)

    try:
        if chart_type == "line":
            x_col = datetime_cols[0] if datetime_cols else cols[0]
            y_cols = numeric_cols if numeric_cols else [cols[-1]]
            data = [
                {
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": chart_df[x_col].astype(str).tolist(),
                    "y": chart_df[y].tolist(),
                    "name": y,
                    "line": {"width": 2.5},
                    "marker": {"size": 5},
                }
                for y in y_cols[:5]
            ]
            layout = _base_layout(title or f"{' & '.join(y_cols)} over {x_col}")
            layout["xaxis"] = {"title": x_col}
            layout["yaxis"] = {"title": y_cols[0] if y_cols else "Value"}

        elif chart_type == "bar":
            x_col = category_cols[0] if category_cols else cols[0]
            y_col = numeric_cols[0] if numeric_cols else cols[-1]
            # Sort by y descending
            sorted_df = chart_df.sort_values(y_col, ascending=False).head(20)
            data = [{
                "type": "bar",
                "x": sorted_df[x_col].astype(str).tolist(),
                "y": sorted_df[y_col].tolist(),
                "marker": {"color": sorted_df[y_col].tolist(), "colorscale": "Teal"},
            }]
            layout = _base_layout(title or f"{y_col} by {x_col}")
            layout["xaxis"] = {"title": x_col, "tickangle": -30}
            layout["yaxis"] = {"title": y_col}

        elif chart_type == "pie":
            label_col = category_cols[0] if category_cols else cols[0]
            value_col = numeric_cols[0] if numeric_cols else cols[-1]
            data = [{
                "type": "pie",
                "labels": chart_df[label_col].astype(str).tolist(),
                "values": chart_df[value_col].tolist(),
                "hole": 0.35,
                "textinfo": "percent+label",
            }]
            layout = _base_layout(title or f"{value_col} distribution")

        elif chart_type == "scatter":
            x_col = numeric_cols[0] if len(numeric_cols) >= 1 else cols[0]
            y_col = numeric_cols[1] if len(numeric_cols) >= 2 else cols[-1]
            color_col = category_cols[0] if category_cols else None
            data = [{
                "type": "scatter",
                "mode": "markers",
                "x": chart_df[x_col].tolist(),
                "y": chart_df[y_col].tolist(),
                "text": chart_df[color_col].astype(str).tolist() if color_col else None,
                "marker": {
                    "size": 8,
                    "opacity": 0.75,
                    "color": chart_df[color_col].astype("category").cat.codes.tolist()
                    if color_col else "#4ECDC4",
                    "colorscale": "Viridis" if color_col else None,
                },
            }]
            layout = _base_layout(title or f"{x_col} vs {y_col}")
            layout["xaxis"] = {"title": x_col}
            layout["yaxis"] = {"title": y_col}

        else:
            # Table view
            return _table_config(chart_df, title)

        return {"type": chart_type, "data": data, "layout": layout}

    except Exception as e:
        logger.error(f"Chart build error: {e}")
        return _table_config(chart_df, title)


def _base_layout(title: str) -> dict:
    return {
        "title": {
            "text": title,
            "font": {"size": 15, "family": "Inter, system-ui, sans-serif"},
        },
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, system-ui, sans-serif", "size": 12},
        "margin": {"t": 50, "l": 60, "r": 20, "b": 60},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": -0.3},
        "hovermode": "closest",
        "colorway": ["#4ECDC4", "#FF6B6B", "#45B7D1", "#96CEB4", "#FFEAA7"],
    }


def _empty_chart(title: str) -> dict:
    return {
        "type": "empty",
        "data": [],
        "layout": _base_layout(title or "No data"),
    }


def _table_config(df: pd.DataFrame, title: str) -> dict:
    """Return a Plotly table config."""
    data = [{
        "type": "table",
        "header": {
            "values": list(df.columns),
            "fill": {"color": "#1a1a2e"},
            "font": {"color": "white", "size": 12},
        },
        "cells": {
            "values": [df[col].astype(str).tolist() for col in df.columns],
            "fill": {"color": ["#f8f9fa", "#ffffff"]},
            "font": {"size": 11},
        },
    }]
    layout = _base_layout(title)
    return {"type": "table", "data": data, "layout": layout}


# ─────────────────────────────────────────────────────────────────────────────
# Forecast chart
# ─────────────────────────────────────────────────────────────────────────────

def build_forecast_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str = "Forecast",
) -> dict:
    """Build a combined historical + forecast line chart."""
    data = [
        {
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Historical",
            "x": historical_df[date_col].astype(str).tolist(),
            "y": historical_df[value_col].tolist(),
            "line": {"color": "#4ECDC4", "width": 2},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Forecast",
            "x": forecast_df["ds"].astype(str).tolist(),
            "y": forecast_df["yhat"].tolist(),
            "line": {"color": "#FF6B6B", "width": 2, "dash": "dot"},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Upper bound",
            "x": forecast_df["ds"].astype(str).tolist(),
            "y": forecast_df["yhat_upper"].tolist(),
            "line": {"color": "#FF6B6B", "width": 0},
            "fill": "tonexty",
            "fillcolor": "rgba(255,107,107,0.1)",
            "showlegend": False,
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Lower bound",
            "x": forecast_df["ds"].astype(str).tolist(),
            "y": forecast_df["yhat_lower"].tolist(),
            "line": {"color": "#FF6B6B", "width": 0},
            "showlegend": False,
        },
    ]

    layout = _base_layout(title)
    layout["xaxis"] = {"title": "Date"}
    layout["yaxis"] = {"title": value_col}

    return {"type": "line", "data": data, "layout": layout}
