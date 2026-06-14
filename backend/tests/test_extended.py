"""
Extended Test Suite — Phases 11-17
Covers: memory, cleaning, MCP tools, agent orchestrator, observability API.
Run with: pytest tests/ -v -m "not slow"
"""

import pytest
import pandas as pd
import numpy as np
import json
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures (augments conftest.py)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sales_df():
    np.random.seed(0)
    n = 300
    return pd.DataFrame({
        "order_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "customer":   np.random.choice(["Alice", "Bob", "Carol"], n),
        "product":    np.random.choice(["Laptop", "Phone", "Tablet"], n),
        "region":     np.random.choice(["North", "South", "East", "West"], n),
        "revenue":    np.round(np.random.uniform(100, 5000, n), 2),
        "quantity":   np.random.randint(1, 10, n),
        "status":     np.random.choice(["Delivered", "Pending", "Cancelled"], n),
    })


@pytest.fixture
def dirty_df():
    """DataFrame with realistic data quality issues."""
    df = pd.DataFrame({
        "name":    ["Alice", "Bob", None, "Dave", "Eve", "Bob", None],
        "revenue": [1000.0, 2000.0, None, 4000.0, None, 2000.0, 500.0],
        "city":    ["Pune ", " Mumbai", "Delhi", "Pune ", "Mumbai", " Mumbai", "Delhi"],
        "score":   [85, 90, 78, None, 92, 90, 88],
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Phase 11 — Web Search
# ─────────────────────────────────────────────────────────────────────────────

class TestWebSearch:

    @pytest.mark.asyncio
    async def test_search_returns_list(self):
        from app.services.web_search import search_web
        results = await search_web("test query", max_results=3)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_result_structure(self):
        from app.services.web_search import search_web
        results = await search_web("revenue trends 2024")
        assert len(results) > 0
        r = results[0]
        assert "title" in r
        assert "url" in r
        assert "content" in r

    @pytest.mark.asyncio
    async def test_web_synthesis_format(self):
        from app.services.llm_service import synthesise_web_results
        internal = "Revenue grew 12% in Q3."
        web = [{"title": "Market Report", "url": "https://example.com", "content": "Industry grew 8%."}]
        result = await synthesise_web_results("Why did revenue grow?", internal, web)
        assert isinstance(result, str)
        assert len(result) > 10


# ─────────────────────────────────────────────────────────────────────────────
# Phase 12 — Memory System
# ─────────────────────────────────────────────────────────────────────────────

class TestMemory:

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        from app.services.memory import store_memory, get_conversation_history
        await store_memory(
            session_id="test-mem-session",
            question="What is total revenue?",
            classification="DATABASE",
            sql="SELECT SUM(revenue) FROM main",
            result_preview="total_revenue\n12345.67",
            insight="Revenue is performing well.",
        )
        history = await get_conversation_history("test-mem-session", limit=10)
        # conftest mocks return empty list — verify call doesn't raise
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_get_conversation_history_empty(self):
        from app.services.memory import get_conversation_history
        history = await get_conversation_history("nonexistent-session", limit=5)
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_clear_session_memory(self):
        from app.services.memory import clear_session_memory
        deleted = await clear_session_memory("test-session-clear")
        assert isinstance(deleted, int)

    @pytest.mark.asyncio
    async def test_get_session_summary(self):
        from app.services.memory import get_session_summary
        summary = await get_session_summary("test-session")
        assert "total_questions" in summary
        assert "database_queries" in summary
        assert "web_queries" in summary

    @pytest.mark.asyncio
    async def test_search_memory(self):
        from app.services.memory import search_memory
        results = await search_memory("test-session", "revenue")
        assert isinstance(results, list)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 13 — Agent Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentOrchestrator:

    @pytest.mark.asyncio
    async def test_database_question(self, sales_df, tmp_path, monkeypatch):
        from app.services.ingestion import save_session_data
        from app.agents.orchestrator import run_agent

        sid = "orch-test-01"
        monkeypatch.setattr("app.core.config.settings.UPLOAD_DIR", str(tmp_path))
        save_session_data(sales_df, sid, "main")

        response = await run_agent(
            question="Show top 5 customers by revenue",
            session_id=sid,
            dataframes={"main": sales_df},
        )
        assert response.classification in ("DATABASE", "WEB", "HYBRID")
        assert isinstance(response.plan, list)
        assert len(response.plan) > 0
        assert response.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_forecast_detection(self, sales_df, tmp_path, monkeypatch):
        from app.services.ingestion import save_session_data
        from app.agents.orchestrator import run_agent

        sid = "orch-test-02"
        monkeypatch.setattr("app.core.config.settings.UPLOAD_DIR", str(tmp_path))
        save_session_data(sales_df, sid, "main")

        response = await run_agent(
            question="Forecast revenue for next 3 months",
            session_id=sid,
            dataframes={"main": sales_df},
        )
        assert response.is_forecast is True

    @pytest.mark.asyncio
    async def test_ambiguity_returns_options(self, sales_df, monkeypatch):
        from app.agents.orchestrator import run_agent

        # Override ambiguity check to return ambiguous
        async def mock_ambiguous(q, s):
            return {
                "is_ambiguous": True,
                "ambiguity_reason": "Could mean two things",
                "options": [
                    {"label": "By revenue", "rewritten_question": "top customers by revenue"},
                    {"label": "By count", "rewritten_question": "top customers by order count"},
                ],
            }
        monkeypatch.setattr("app.agents.orchestrator.check_ambiguity", mock_ambiguous)

        response = await run_agent(
            question="Show best customers",
            session_id="orch-test-03",
            dataframes={"main": sales_df},
        )
        assert response.is_ambiguous is True
        assert len(response.clarification_options) == 2

    @pytest.mark.asyncio
    async def test_response_dict_has_all_keys(self, sales_df):
        from app.agents.orchestrator import run_agent

        response = await run_agent(
            question="Show total revenue",
            session_id="orch-test-04",
            dataframes={"main": sales_df},
        )
        d = response.to_dict()
        expected_keys = [
            "question", "classification", "is_ambiguous", "clarification_options",
            "plan", "sql", "sql_validation_error", "query_result", "chart",
            "chart_type", "insight", "web_results", "is_forecast", "forecast_data",
            "error", "execution_steps", "total_time_ms",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key in response: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Data Cleaning Service
# ─────────────────────────────────────────────────────────────────────────────

class TestCleaningService:

    def test_auto_clean_removes_duplicates(self, dirty_df):
        from app.services.cleaning import clean_dataframe
        cleaned, report = clean_dataframe(dirty_df, strategy="auto")
        assert cleaned.duplicated().sum() == 0
        assert report["rows_removed"] >= 1

    def test_auto_clean_imputes_numeric(self, dirty_df):
        from app.services.cleaning import clean_dataframe
        cleaned, report = clean_dataframe(dirty_df, strategy="auto")
        # revenue and score had nulls — should be filled
        assert cleaned["revenue"].isna().sum() == 0 or report["missing_remaining"] < dirty_df.isna().sum().sum()

    def test_conservative_strategy_no_imputation(self, dirty_df):
        from app.services.cleaning import clean_dataframe
        cleaned, report = clean_dataframe(dirty_df, strategy="conservative")
        assert report["values_imputed"] == 0

    def test_aggressive_strategy_drops_all_nulls(self):
        from app.services.cleaning import clean_dataframe
        df = pd.DataFrame({
            "a": [1.0, None, 3.0, 4.0],
            "b": ["x", "y", None, "w"],
        })
        cleaned, report = clean_dataframe(df, strategy="aggressive")
        assert cleaned.isna().sum().sum() == 0

    def test_whitespace_trimming(self, dirty_df):
        from app.services.cleaning import clean_dataframe
        cleaned, report = clean_dataframe(dirty_df, strategy="conservative")
        # "Pune " should become "Pune"
        assert "Pune " not in cleaned["city"].values

    def test_custom_rules_applied(self):
        from app.services.cleaning import clean_dataframe
        # Mixed nulls: no row is fully empty so drop-fully-empty does not remove them
        df = pd.DataFrame({
            "sales":    [100.0, None,  300.0, 150.0, 200.0],
            "category": ["A",   "B",   None,  "D",   "C"  ],
        })
        cleaned, report = clean_dataframe(
            df,
            strategy="auto",
            custom_rules={"sales": "zero", "category": "unknown"},
        )
        assert cleaned["sales"].isna().sum() == 0, "sales nulls should be filled with 0"
        assert "Unknown" in cleaned["category"].tolist(), "Unknown should appear in category"

    def test_report_structure(self, dirty_df):
        from app.services.cleaning import clean_dataframe
        _, report = clean_dataframe(dirty_df)
        assert "original_rows" in report
        assert "operations" in report
        assert "final_rows" in report
        assert "missing_remaining" in report
        assert isinstance(report["operations"], list)

    def test_no_operations_for_clean_data(self):
        from app.services.cleaning import clean_dataframe
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        _, report = clean_dataframe(df)
        assert report["rows_removed"] == 0
        assert report["values_imputed"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Outlier Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestOutlierDetection:

    def test_iqr_detects_outliers(self):
        from app.services.cleaning import detect_outliers
        vals = list(range(1, 51)) + [1000, -500]
        df = pd.DataFrame({"revenue": vals})
        result = detect_outliers(df, method="iqr")
        assert "revenue" in result
        assert result["revenue"]["outlier_count"] >= 2

    def test_zscore_detects_outliers(self):
        from app.services.cleaning import detect_outliers
        vals = list(range(1, 51)) + [9999]
        df = pd.DataFrame({"score": vals})
        result = detect_outliers(df, method="zscore")
        assert "score" in result

    def test_no_outliers_in_uniform_data(self):
        from app.services.cleaning import detect_outliers
        df = pd.DataFrame({"value": [10.0] * 50})
        result = detect_outliers(df)
        assert result == {}

    def test_skips_non_numeric_columns(self):
        from app.services.cleaning import detect_outliers
        df = pd.DataFrame({"name": ["Alice"] * 20, "score": list(range(20))})
        result = detect_outliers(df)
        assert "name" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 16 — MCP Tools
# ─────────────────────────────────────────────────────────────────────────────

class TestMCPTools:

    def test_tool_list_completeness(self):
        from app.tools.mcp_tools import MCP_TOOLS
        names = {t["name"] for t in MCP_TOOLS}
        required = {"search_schema", "run_sql", "search_web", "get_history", "forecast", "generate_chart"}
        assert required.issubset(names), f"Missing tools: {required - names}"

    def test_each_tool_has_required_fields(self):
        from app.tools.mcp_tools import MCP_TOOLS
        for tool in MCP_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool
            assert isinstance(tool["description"], str) and len(tool["description"]) > 10

    @pytest.mark.asyncio
    async def test_run_sql_tool(self, sales_df):
        from app.tools.mcp_tools import MCPToolExecutor
        executor = MCPToolExecutor(session_id="mcp-test", dataframes={"main": sales_df})
        result = await executor.execute("run_sql", {
            "sql": "SELECT COUNT(*) AS cnt FROM main"
        })
        assert "row_count" in result or "columns" in result

    @pytest.mark.asyncio
    async def test_run_sql_tool_blocks_delete(self, sales_df):
        from app.tools.mcp_tools import MCPToolExecutor
        executor = MCPToolExecutor(session_id="mcp-test", dataframes={"main": sales_df})
        result = await executor.execute("run_sql", {"sql": "DELETE FROM main"})
        assert "error" in result
        assert result.get("blocked") is True

    @pytest.mark.asyncio
    async def test_search_web_tool(self, sales_df):
        from app.tools.mcp_tools import MCPToolExecutor
        executor = MCPToolExecutor(session_id="mcp-test", dataframes={"main": sales_df})
        result = await executor.execute("search_web", {"query": "market trends 2024"})
        assert "results" in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_generate_chart_tool(self, sales_df):
        from app.tools.mcp_tools import MCPToolExecutor
        executor = MCPToolExecutor(session_id="mcp-test", dataframes={"main": sales_df})
        agg = sales_df.groupby("region")["revenue"].sum().reset_index()
        result = await executor.execute("generate_chart", {
            "data": {
                "columns": list(agg.columns),
                "data": agg.values.tolist(),
            },
            "chart_type": "bar",
            "title": "Revenue by Region",
        })
        assert "type" in result
        assert "data" in result

    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self, sales_df):
        from app.tools.mcp_tools import MCPToolExecutor
        executor = MCPToolExecutor(session_id="mcp-test", dataframes={"main": sales_df})
        result = await executor.execute("nonexistent_tool", {})
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 17 — Observability
# ─────────────────────────────────────────────────────────────────────────────

class TestObservability:

    def test_timer_context_manager(self):
        import time
        from app.services.observability import Timer
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed_ms >= 40  # allow some slack

    def test_timer_measures_correctly(self):
        from app.services.observability import Timer
        with Timer() as t:
            x = sum(range(100_000))
        assert t.elapsed_ms > 0
        assert t.elapsed_ms < 5000  # should be well under 5 seconds

    @pytest.mark.asyncio
    async def test_log_query_does_not_raise(self, monkeypatch):
        """log_query should work even if Dynatrace is unavailable."""
        from app.services.observability import log_query
        from unittest.mock import AsyncMock, MagicMock

        mock_db = MagicMock()
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        # Patch so no real DB write happens
        import app.services.observability as obs_module
        original_querylog = obs_module.QueryLog

        class FakeQueryLog:
            def __init__(self, **kwargs):
                self.id = 1
                for k, v in kwargs.items():
                    setattr(self, k, v)

        obs_module.QueryLog = FakeQueryLog
        try:
            log_id = await log_query(
                db=mock_db,
                session_id="obs-test",
                question="test question",
                classification="DATABASE",
                total_time_ms=123.4,
            )
            assert isinstance(log_id, int) or log_id is not None
        finally:
            obs_module.QueryLog = original_querylog


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — RAG Store
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGStore:

    def test_hash_embedding_returns_correct_dims(self):
        from app.services.rag_store import _hash_embedding
        embedding = _hash_embedding("test schema text for sales table", dims=384)
        assert len(embedding) == 384
        assert all(isinstance(v, float) for v in embedding)

    def test_hash_embedding_is_deterministic(self):
        from app.services.rag_store import _hash_embedding
        e1 = _hash_embedding("same text", dims=384)
        e2 = _hash_embedding("same text", dims=384)
        assert e1 == e2

    def test_hash_embedding_differs_for_different_text(self):
        from app.services.rag_store import _hash_embedding
        e1 = _hash_embedding("sales revenue data", dims=384)
        e2 = _hash_embedding("employee HR records", dims=384)
        assert e1 != e2

    @pytest.mark.asyncio
    async def test_es_health_returns_dict(self):
        from app.services.rag_store import es_health
        result = await es_health()
        assert isinstance(result, dict)
        assert "status" in result

    @pytest.mark.asyncio
    async def test_semantic_search_returns_list_without_es(self):
        """Should return empty list gracefully when ES is unavailable."""
        from app.services.rag_store import semantic_search
        results = await semantic_search("revenue by region", "test-session", top_k=3)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_index_schema_graceful_without_es(self):
        from app.services.rag_store import index_schema
        schema_doc = {
            "session_id": "test",
            "table_name": "sales",
            "rag_text": "Table: sales\nColumn: revenue (currency)",
            "columns": [{"name": "revenue", "type": "currency"}],
            "row_count": 100,
        }
        result = await index_schema(schema_doc)
        # Should return False gracefully (ES not available in test env)
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Service — additional coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMService:

    @pytest.mark.asyncio
    async def test_classify_returns_valid_category(self):
        from app.services.llm_service import classify_question
        schema = "Table: sales\n- revenue (currency)\n- order_date (datetime)"
        result = await classify_question("What is total revenue?", schema)
        assert result in ("DATABASE", "WEB", "HYBRID")

    @pytest.mark.asyncio
    async def test_generate_sql_returns_select(self):
        from app.services.llm_service import generate_sql
        schema = "Table: main\n- customer (text)\n- revenue (currency)"
        sql = await generate_sql("top customers by revenue", schema)
        assert "SELECT" in sql.upper()

    @pytest.mark.asyncio
    async def test_check_ambiguity_returns_dict(self):
        from app.services.llm_service import check_ambiguity
        result = await check_ambiguity("show best customers", "Table: main\n- revenue: currency")
        assert "is_ambiguous" in result

    @pytest.mark.asyncio
    async def test_generate_insight_returns_string(self):
        from app.services.llm_service import generate_insight
        result = await generate_insight(
            "What is revenue by region?",
            "SELECT region, SUM(revenue) FROM main GROUP BY region",
            "North: 50000\nSouth: 30000",
            "Table: main\n- region (category)\n- revenue (currency)",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_plan_returns_list(self):
        from app.services.llm_service import generate_plan
        result = await generate_plan(
            "Why did revenue drop in Q3?",
            "Table: main\n- revenue (currency)\n- order_date (datetime)",
            "HYBRID",
            False,
        )
        assert isinstance(result, list)
        assert len(result) >= 1
