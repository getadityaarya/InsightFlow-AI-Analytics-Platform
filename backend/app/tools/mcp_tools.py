"""
Phase 16 — MCP Tools
Exposes core agent capabilities as callable tools so Gemini (or any MCP-compatible
model) can invoke them programmatically during agentic execution.

Each tool is a thin wrapper around the underlying service function.
Gemini decides which tools to call based on the user question.
"""

import json
import pandas as pd
from typing import Any
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions (MCP-compatible schema)
# ─────────────────────────────────────────────────────────────────────────────

MCP_TOOLS = [
    {
        "name": "search_schema",
        "description": (
            "Search the uploaded dataset schema to find relevant columns, "
            "tables, and their descriptions. Use before generating SQL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query to search schema (e.g. 'revenue columns')"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "run_sql",
        "description": (
            "Execute a validated SELECT SQL query against the uploaded dataset. "
            "Only SELECT statements are allowed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL SELECT statement to execute"
                }
            },
            "required": ["sql"]
        }
    },
    {
        "name": "search_web",
        "description": (
            "Search the web for external context, market data, or industry benchmarks. "
            "Use only when the question requires information not in the dataset."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Web search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_history",
        "description": (
            "Retrieve the conversation history for the current session. "
            "Useful for follow-up questions referencing past queries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of past turns to retrieve",
                    "default": 5
                }
            }
        }
    },
    {
        "name": "forecast",
        "description": (
            "Run a time-series forecast using Prophet on a numeric column. "
            "Requires a date column and a value column in the dataset."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date_column": {
                    "type": "string",
                    "description": "Name of the date/datetime column"
                },
                "value_column": {
                    "type": "string",
                    "description": "Name of the numeric column to forecast"
                },
                "periods": {
                    "type": "integer",
                    "description": "Number of future periods to forecast",
                    "default": 30
                },
                "freq": {
                    "type": "string",
                    "description": "Frequency: D (daily), W (weekly), M (monthly), Q (quarterly)",
                    "default": "D"
                }
            },
            "required": ["date_column", "value_column"]
        }
    },
    {
        "name": "generate_chart",
        "description": (
            "Generate a chart from query results. Auto-selects chart type "
            "based on data shape, or use the chart_type parameter."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "description": "Query result dict with 'columns' and 'data' keys"
                },
                "chart_type": {
                    "type": "string",
                    "description": "Optional: line, bar, pie, scatter, table",
                    "enum": ["line", "bar", "pie", "scatter", "table", "auto"]
                },
                "title": {
                    "type": "string",
                    "description": "Chart title"
                }
            },
            "required": ["data"]
        }
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool executor
# ─────────────────────────────────────────────────────────────────────────────

class MCPToolExecutor:
    """
    Dispatches tool calls from the LLM to the appropriate service.
    Used by the agent orchestrator when operating in full agentic mode.
    """

    def __init__(self, session_id: str, dataframes: dict[str, pd.DataFrame]):
        self.session_id = session_id
        self.dataframes = dataframes

    async def execute(self, tool_name: str, parameters: dict) -> Any:
        """Dispatch a tool call and return the result."""
        logger.info(f"MCP tool call: {tool_name}({json.dumps(parameters)[:100]})")

        if tool_name == "search_schema":
            return await self._search_schema(parameters["query"])

        elif tool_name == "run_sql":
            return await self._run_sql(parameters["sql"])

        elif tool_name == "search_web":
            return await self._search_web(
                parameters["query"],
                parameters.get("max_results", 5)
            )

        elif tool_name == "get_history":
            return await self._get_history(parameters.get("limit", 5))

        elif tool_name == "forecast":
            return await self._forecast(parameters)

        elif tool_name == "generate_chart":
            return await self._generate_chart(parameters)

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def _search_schema(self, query: str) -> dict:
        from app.services.schema_intelligence import build_sql_context
        context = await build_sql_context(self.session_id)
        return {"schema_context": context, "query": query}

    async def _run_sql(self, sql: str) -> dict:
        from app.services.sql_engine import validate_sql, execute_sql_on_dataframes, SQLValidationError
        try:
            validated = validate_sql(sql)
            result = execute_sql_on_dataframes(validated, self.dataframes, self.session_id)
            return result
        except SQLValidationError as e:
            return {"error": str(e), "blocked": True}

    async def _search_web(self, query: str, max_results: int) -> dict:
        from app.services.web_search import search_web
        results = await search_web(query, max_results)
        return {"results": results, "count": len(results)}

    async def _get_history(self, limit: int) -> dict:
        from app.services.memory import get_conversation_history
        history = await get_conversation_history(self.session_id, limit)
        return {"history": history, "count": len(history)}

    async def _forecast(self, params: dict) -> dict:
        from app.services.forecasting import run_forecast, ForecastError
        if not self.dataframes:
            return {"error": "No data loaded"}
        df = next(iter(self.dataframes.values()))
        try:
            result = run_forecast(
                df,
                params["date_column"],
                params["value_column"],
                params.get("periods", 30),
                params.get("freq", "D"),
            )
            return {
                "summary": result["summary"],
                "confidence_interval": result["confidence_interval"],
                "trend_pct": result["trend_pct"],
                "direction": result["direction"],
            }
        except ForecastError as e:
            return {"error": str(e)}

    async def _generate_chart(self, params: dict) -> dict:
        from app.services.chart_engine import select_chart_type, build_plotly_config
        data = params["data"]
        df = pd.DataFrame(data.get("data", []), columns=data.get("columns", []))
        chart_type = params.get("chart_type", "auto")
        if chart_type == "auto":
            chart_type = select_chart_type(df)
        config = build_plotly_config(df, chart_type, params.get("title", ""))
        return config
