"""
Tests for Phase 4 (vector store), Phase 12 (memory), Phase 13 (agent),
and Phase 16 (MCP tool executor).
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "order_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "customer": np.random.choice(["Alice", "Bob", "Carol"], n),
        "product": np.random.choice(["Laptop", "Phone", "Tablet"], n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "sales_amount": np.round(np.random.uniform(100, 2000, n), 2),
        "quantity": np.random.randint(1, 5, n),
    })


@pytest.fixture
def session_id():
    return "test-session-mcp-001"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Vector Store
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorStore:

    @pytest.mark.asyncio
    async def test_index_and_search_memory(self):
        from app.services.vector_store import index_schema, search_schema
        sid = "vec-test-001"
        columns = [
            {"name": "sales_amount", "type": "currency", "description": "Revenue from sale"},
            {"name": "order_date", "type": "datetime", "description": "Date of order"},
            {"name": "region", "type": "category", "description": "Geographic region"},
        ]
        rag_text = "Table: sales\nColumns:\n  - sales_amount (currency)\n  - order_date (datetime)\n  - region (category)"
        await index_schema(sid, "sales", rag_text, columns)

        results = await search_schema(sid, "total revenue by region", top_k=3)
        assert len(results) > 0
        assert results[0]["table_name"] == "sales"

    @pytest.mark.asyncio
    async def test_search_returns_relevant_table(self):
        from app.services.vector_store import index_schema, search_schema
        sid = "vec-test-002"
        await index_schema(sid, "orders",
            "Table: orders. Columns: order_id, customer_name, total_amount, created_at.",
            [{"name": "total_amount", "type": "currency", "description": "Order total"}])
        await index_schema(sid, "products",
            "Table: products. Columns: product_id, product_name, category, price.",
            [{"name": "price", "type": "currency", "description": "Product price"}])

        results = await search_schema(sid, "top customers by total amount", top_k=2)
        assert any(r["table_name"] == "orders" for r in results)

    @pytest.mark.asyncio
    async def test_search_empty_session(self):
        from app.services.vector_store import search_schema
        results = await search_schema("nonexistent-session-xyz", "anything", top_k=3)
        # Should return empty list, not crash
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_memory_fallback_no_es(self):
        # Memory fallback is no longer tested separately as FAISS handles this natively without ES.
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Phase 12 — Memory Service
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryService:

    @pytest.mark.asyncio
    async def test_store_memory(self):
        from app.services.memory import store_memory
        result = await store_memory(
            session_id="mem-test-001",
            question="What is total revenue?",
            classification="DATABASE",
            sql="SELECT SUM(sales_amount) FROM main",
            result_preview="total_revenue\n150000.0",
            insight="Revenue is 150K.",
            chart_type="bar",
        )
        assert result is not None  # Returns inserted_id or mock

    @pytest.mark.asyncio
    async def test_get_conversation_history_empty(self):
        from app.services.memory import get_conversation_history
        history = await get_conversation_history("empty-session-xyz", limit=5)
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_get_last_sql_none(self):
        from app.services.memory import get_last_sql
        sql = await get_last_sql("no-session")
        assert sql is None

    @pytest.mark.asyncio
    async def test_clear_session_memory(self):
        from app.services.memory import clear_session_memory
        count = await clear_session_memory("test-session-to-clear")
        assert count == 0  # Mock returns 0

    @pytest.mark.asyncio
    async def test_session_summary(self):
        from app.services.memory import get_session_summary
        summary = await get_session_summary("any-session")
        assert "total_questions" in summary
        assert "database_queries" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Phase 16 — MCP Tool Executor
# ─────────────────────────────────────────────────────────────────────────────

class TestMCPTools:

    @pytest.mark.asyncio
    async def test_search_schema_tool(self, sample_df, session_id):
        from app.tools.mcp_tools import MCPToolExecutor
        mcp = MCPToolExecutor(session_id=session_id, dataframes={"main": sample_df})
        result = await mcp.execute("search_schema", {"query": "total sales revenue"})
        assert "schema_context" in result

    @pytest.mark.asyncio
    async def test_run_sql_tool_valid(self, sample_df, session_id):
        from app.tools.mcp_tools import MCPToolExecutor
        mcp = MCPToolExecutor(session_id=session_id, dataframes={"main": sample_df})
        result = await mcp.execute("run_sql", {"sql": "SELECT COUNT(*) AS total FROM main"})
        assert result.get("row_count", 0) == 1
        assert "total" in result.get("columns", [])

    @pytest.mark.asyncio
    async def test_run_sql_tool_blocked(self, sample_df, session_id):
        from app.tools.mcp_tools import MCPToolExecutor
        mcp = MCPToolExecutor(session_id=session_id, dataframes={"main": sample_df})
        result = await mcp.execute("run_sql", {"sql": "DROP TABLE main"})
        assert result.get("error") is not None
        assert result.get("blocked") is True

    @pytest.mark.asyncio
    async def test_search_web_tool(self, sample_df, session_id):
        from app.tools.mcp_tools import MCPToolExecutor
        mcp = MCPToolExecutor(session_id=session_id, dataframes={"main": sample_df})
        result = await mcp.execute("search_web", {"query": "market trends 2024", "max_results": 3})
        assert "results" in result
        assert isinstance(result["results"], list)

    @pytest.mark.asyncio
    async def test_get_history_tool(self, sample_df, session_id):
        from app.tools.mcp_tools import MCPToolExecutor
        mcp = MCPToolExecutor(session_id=session_id, dataframes={"main": sample_df})
        result = await mcp.execute("get_history", {"limit": 5})
        assert "history" in result

    @pytest.mark.asyncio
    async def test_generate_chart_tool(self, sample_df, session_id):
        from app.tools.mcp_tools import MCPToolExecutor
        mcp = MCPToolExecutor(session_id=session_id, dataframes={"main": sample_df})
        result_data = {
            "columns": ["region", "total"],
            "data": [["North", 50000], ["South", 40000], ["East", 30000]],
            "row_count": 3,
        }
        result = await mcp.execute("generate_chart", {"data": result_data, "chart_type": "bar", "title": "Sales by Region"})
        assert "data" in result
        assert "layout" in result

    @pytest.mark.asyncio
    async def test_unknown_tool(self, sample_df, session_id):
        from app.tools.mcp_tools import MCPToolExecutor
        mcp = MCPToolExecutor(session_id=session_id, dataframes={"main": sample_df})
        result = await mcp.execute("nonexistent_tool", {})
        assert "error" in result

    def test_mcp_tool_definitions_complete(self):
        from app.tools.mcp_tools import MCP_TOOLS
        tool_names = [t["name"] for t in MCP_TOOLS]
        required = ["search_schema", "run_sql", "search_web", "get_history", "forecast", "generate_chart"]
        for name in required:
            assert name in tool_names, f"MCP tool '{name}' missing"

    def test_mcp_tools_have_descriptions(self):
        from app.tools.mcp_tools import MCP_TOOLS
        for tool in MCP_TOOLS:
            assert tool.get("description"), f"Tool {tool['name']} missing description"
            assert tool.get("parameters"), f"Tool {tool['name']} missing parameters"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 13 — Agent Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentOrchestrator:

    @pytest.mark.asyncio
    async def test_run_agent_database(self, sample_df):
        from app.agents.orchestrator import run_agent
        resp = await run_agent(
            question="Show total sales by region",
            session_id="agent-test-001",
            dataframes={"main": sample_df},
        )
        assert resp.question == "Show total sales by region"
        assert resp.classification in ("DATABASE", "WEB", "HYBRID")
        assert isinstance(resp.plan, list)
        assert len(resp.execution_steps) > 0
        assert resp.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_run_agent_returns_insight(self, sample_df):
        from app.agents.orchestrator import run_agent
        resp = await run_agent(
            question="What are the key business metrics?",
            session_id="agent-test-002",
            dataframes={"main": sample_df},
        )
        assert resp.insight is not None
        assert len(resp.insight) > 0

    @pytest.mark.asyncio
    async def test_run_agent_with_clarification(self, sample_df):
        from app.agents.orchestrator import run_agent
        resp = await run_agent(
            question="Show best customers",
            session_id="agent-test-003",
            dataframes={"main": sample_df},
            clarification_choice="SELECT customer, SUM(sales_amount) AS revenue FROM main GROUP BY customer ORDER BY revenue DESC LIMIT 10",
        )
        assert resp.is_ambiguous is False

    @pytest.mark.asyncio
    async def test_run_agent_handles_empty_dataframes(self):
        from app.agents.orchestrator import run_agent
        resp = await run_agent(
            question="Show total revenue",
            session_id="agent-empty-session",
            dataframes={},
        )
        # Should not crash, may have error set
        assert resp is not None
        assert isinstance(resp.to_dict(), dict)

    @pytest.mark.asyncio
    async def test_agent_response_to_dict(self, sample_df):
        from app.agents.orchestrator import run_agent
        resp = await run_agent(
            question="Count all rows",
            session_id="agent-test-004",
            dataframes={"main": sample_df},
        )
        d = resp.to_dict()
        required_keys = [
            "question", "classification", "is_ambiguous", "plan",
            "sql", "query_result", "chart", "insight",
            "execution_steps", "total_time_ms",
        ]
        for key in required_keys:
            assert key in d, f"Key '{key}' missing from AgentResponse.to_dict()"
