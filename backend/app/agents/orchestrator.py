"""
Phase 13 — Agent Orchestrator (LangGraph-style state machine)

Full agentic pipeline:
  State → Classify → Ambiguity → Plan (LLM) → Tool execution → Synthesise → Memory

Phase 16 MCP tools are fully wired — the LLM plan selects which tools to call.
Now powered by LangGraph for State management and routing.
"""

import logging
import time
import traceback
import operator
from typing import Optional, Annotated, TypedDict
import pandas as pd

from langgraph.graph import StateGraph, END

from app.services.llm_service import (
    classify_question, generate_sql, generate_insight,
    check_ambiguity, narrate_forecast, synthesise_web_results,
    generate_agent_plan,
)
from app.services.sql_engine import validate_sql, execute_sql_on_dataframes, SQLValidationError
from app.services.schema_intelligence import build_sql_context
from app.services.chart_engine import select_chart_type, build_plotly_config, build_forecast_chart
from app.services.memory import store_memory, get_conversation_history
from app.services.forecasting import (
    is_forecast_question, parse_forecast_periods,
    run_forecast, detect_forecast_columns, ForecastError,
)
from app.services.web_search import search_web
from app.tools.mcp_tools import MCPToolExecutor

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    dataframes: dict
    clarification_choice: Optional[str]
    schema_context: str
    classification: str
    is_ambiguous: bool
    clarification_options: list
    plan: list[str]
    sql: Optional[str]
    sql_validation_error: Optional[str]
    query_result: Optional[dict]
    chart: Optional[dict]
    chart_type: Optional[str]
    insight: str
    web_results: list
    is_forecast: bool
    forecast_data: Optional[dict]
    error: Optional[str]
    execution_steps: Annotated[list[dict], operator.add]
    total_time_ms: float


class AgentResponse:
    def __init__(self, state: AgentState):
        self.question = state.get("question", "")
        self.classification = state.get("classification", "DATABASE")
        self.is_ambiguous = state.get("is_ambiguous", False)
        self.clarification_options = state.get("clarification_options", [])
        self.plan = state.get("plan", [])
        self.sql = state.get("sql")
        self.sql_validation_error = state.get("sql_validation_error")
        self.query_result = state.get("query_result")
        self.chart = state.get("chart")
        self.chart_type = state.get("chart_type")
        self.insight = state.get("insight", "")
        self.web_results = state.get("web_results", [])
        self.is_forecast = state.get("is_forecast", False)
        self.forecast_data = state.get("forecast_data")
        self.error = state.get("error")
        self.execution_steps = state.get("execution_steps", [])
        self.total_time_ms = state.get("total_time_ms", 0.0)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _format_result_preview(result: Optional[dict], max_rows: int = 20) -> str:
    if not result or not result.get("data"):
        return "No results"
    df = pd.DataFrame(result["data"][:max_rows], columns=result["columns"])
    return df.to_string(index=False, max_rows=max_rows)


# ─── LangGraph Nodes ──────────────────────────────────────────────────────────

async def retrieve_schema_node(state: AgentState):
    schema_context = await build_sql_context(state["session_id"], query=state["question"])
    return {
        "schema_context": schema_context,
        "execution_steps": [{"step": "RAG schema retrieval", "status": "ok", "detail": f"{len(schema_context)} chars"}]
    }

async def classify_node(state: AgentState):
    classification = await classify_question(state["question"], state["schema_context"])
    return {
        "classification": classification,
        "execution_steps": [{"step": "Classification", "status": "ok", "detail": classification}]
    }

async def ambiguity_node(state: AgentState):
    if state.get("clarification_choice"):
        return {"question": state["clarification_choice"], "is_ambiguous": False}
        
    ambiguity = await check_ambiguity(state["question"], state["schema_context"])
    if ambiguity.get("is_ambiguous"):
        return {
            "is_ambiguous": True, 
            "clarification_options": ambiguity.get("options", []),
            "execution_steps": [{"step": "Ambiguity detected", "status": "ok", "detail": ambiguity.get("ambiguity_reason", "")}]
        }
    return {"is_ambiguous": False}

async def forecast_detect_node(state: AgentState):
    is_forecast = is_forecast_question(state["question"])
    return {"is_forecast": is_forecast}

async def plan_node(state: AgentState):
    plan_steps = await generate_agent_plan(state["question"], state["schema_context"], state["classification"], state["is_forecast"])
    return {
        "plan": plan_steps,
        "execution_steps": [{"step": "Execution plan built", "status": "ok", "detail": " | ".join(plan_steps[:3])}]
    }

async def execute_database_node(state: AgentState):
    mcp = MCPToolExecutor(session_id=state["session_id"], dataframes=state["dataframes"])
    history = await get_conversation_history(state["session_id"], limit=5)
    
    schema_result = await mcp.execute("search_schema", {"query": state["question"]})
    enriched_context = schema_result.get("schema_context", state["schema_context"])
    
    sql = await generate_sql(state["question"], enriched_context, history)
    steps = [{"step": "SQL generated", "status": "ok", "detail": sql[:120]}]
    
    try:
        validated_sql = validate_sql(sql)
    except SQLValidationError as e:
        steps.append({"step": "SQL validation", "status": "error", "detail": str(e)})
        return {"sql": sql, "sql_validation_error": str(e), "insight": f"SQL blocked: {e}", "execution_steps": steps}
        
    steps.append({"step": "SQL validated", "status": "ok", "detail": ""})
    
    result = await mcp.execute("run_sql", {"sql": validated_sql})
    if result.get("error"):
        steps.append({"step": "SQL execution", "status": "error", "detail": result["error"]})
        return {"sql": sql, "sql_validation_error": result["error"], "insight": f"Execution error: {result['error']}", "execution_steps": steps}
        
    steps.append({"step": "SQL executed", "status": "ok", "detail": f"{result['row_count']} rows in {result['execution_time_ms']}ms"})
    
    chart = None
    chart_type = "table"
    if result["row_count"] > 0:
        chart_result = await mcp.execute("generate_chart", {"data": result, "chart_type": "auto", "title": state["question"][:60]})
        chart = chart_result
        chart_type = chart_result.get("type", "table")
        steps.append({"step": "Chart built", "status": "ok", "detail": chart_type})
        
    insight = await generate_insight(state["question"], sql, _format_result_preview(result), enriched_context)
    steps.append({"step": "Insight generated", "status": "ok", "detail": ""})
    
    return {
        "sql": sql,
        "query_result": result,
        "chart": chart,
        "chart_type": chart_type,
        "insight": insight,
        "execution_steps": steps
    }

async def execute_web_node(state: AgentState):
    mcp = MCPToolExecutor(session_id=state["session_id"], dataframes=state["dataframes"])
    web_result = await mcp.execute("search_web", {"query": state["question"], "max_results": 5})
    web_results = web_result.get("results", [])
    steps = [{"step": "Web search", "status": "ok", "detail": f"{len(web_results)} results"}]
    
    insight = await synthesise_web_results(state["question"], "", web_results)
    steps.append({"step": "Web synthesis complete", "status": "ok", "detail": ""})
    
    return {"web_results": web_results, "insight": insight, "execution_steps": steps}

async def execute_hybrid_node(state: AgentState):
    db_update = await execute_database_node(state)
    steps = db_update.get("execution_steps", [])
    
    mcp = MCPToolExecutor(session_id=state["session_id"], dataframes=state["dataframes"])
    web_result = await mcp.execute("search_web", {"query": state["question"], "max_results": 5})
    web_results = web_result.get("results", [])
    steps.append({"step": "Web search (hybrid)", "status": "ok", "detail": f"{len(web_results)} results"})
    
    insight = await synthesise_web_results(state["question"], db_update.get("insight", ""), web_results)
    steps.append({"step": "Hybrid synthesis complete", "status": "ok", "detail": ""})
    
    db_update["web_results"] = web_results
    db_update["insight"] = insight
    db_update["execution_steps"] = steps
    return db_update

async def execute_forecast_node(state: AgentState):
    mcp = MCPToolExecutor(session_id=state["session_id"], dataframes=state["dataframes"])
    steps = []
    
    if not state["dataframes"]:
        return {"error": "No data available for forecasting."}
        
    first_df = next(iter(state["dataframes"].values()))
    forecast_cols = detect_forecast_columns(first_df)
    if not forecast_cols:
        return {"error": "Could not detect date + numeric column pair for forecasting."}
        
    date_col, value_col = forecast_cols
    periods, freq = parse_forecast_periods(state["question"])
    steps.append({"step": "Forecast params", "status": "ok", "detail": f"{date_col} → {value_col}, {periods} {freq}"})
    
    result = await mcp.execute("forecast", {
        "date_column": date_col, "value_column": value_col,
        "periods": periods, "freq": freq,
    })
    
    if result.get("error"):
        steps.append({"step": "Forecast error", "status": "error", "detail": result["error"]})
        return {"error": result["error"], "execution_steps": steps}
        
    steps.append({"step": "Forecast complete", "status": "ok", "detail": result.get("summary", "")})
    
    chart_type = None
    chart = None
    try:
        raw = run_forecast(first_df, date_col, value_col, periods, freq)
        chart_type = "line"
        chart = build_forecast_chart(
            raw["historical"], raw["forecast"],
            date_col="ds", value_col="y",
            title=f"Forecast: {value_col}",
        )
        steps.append({"step": "Forecast chart built", "status": "ok", "detail": ""})
    except Exception as e:
        steps.append({"step": "Forecast chart", "status": "error", "detail": str(e)})
        
    insight = await narrate_forecast(
        state["question"], result.get("summary", ""), result.get("confidence_interval", "")
    )
    steps.append({"step": "Forecast narration complete", "status": "ok", "detail": ""})
    
    return {
        "forecast_data": result,
        "chart_type": chart_type,
        "chart": chart,
        "insight": insight,
        "execution_steps": steps
    }

async def store_memory_node(state: AgentState):
    if not state.get("error"):
        await store_memory(
            session_id=state["session_id"],
            question=state["question"],
            classification=state.get("classification", "DATABASE"),
            sql=state.get("sql"),
            result_preview=_format_result_preview(state.get("query_result")),
            insight=state.get("insight", ""),
            chart_type=state.get("chart_type"),
        )
        return {"execution_steps": [{"step": "Memory stored", "status": "ok", "detail": ""}]}
    return {"execution_steps": [{"step": "Memory skipped", "status": "ok", "detail": "Due to previous error"}]}


# ─── LangGraph Routing ────────────────────────────────────────────────────────

def route_ambiguity(state: AgentState):
    if state.get("is_ambiguous"):
        return END
    return "forecast_detect"

def route_execution(state: AgentState):
    if state.get("is_forecast"):
        return "execute_forecast"
    
    c = state.get("classification", "DATABASE")
    if c == "WEB":
        return "execute_web"
    if c == "HYBRID":
        return "execute_hybrid"
    return "execute_database"

# ─── LangGraph Builder ────────────────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve_schema", retrieve_schema_node)
    workflow.add_node("classify", classify_node)
    workflow.add_node("ambiguity", ambiguity_node)
    workflow.add_node("forecast_detect", forecast_detect_node)
    workflow.add_node("generate_plan", plan_node)
    workflow.add_node("execute_database", execute_database_node)
    workflow.add_node("execute_web", execute_web_node)
    workflow.add_node("execute_hybrid", execute_hybrid_node)
    workflow.add_node("execute_forecast", execute_forecast_node)
    workflow.add_node("store_memory", store_memory_node)

    workflow.set_entry_point("retrieve_schema")
    
    workflow.add_edge("retrieve_schema", "classify")
    workflow.add_edge("classify", "ambiguity")
    workflow.add_conditional_edges("ambiguity", route_ambiguity, {
        END: END,
        "forecast_detect": "forecast_detect"
    })
    
    workflow.add_edge("forecast_detect", "generate_plan")
    workflow.add_conditional_edges("generate_plan", route_execution, {
        "execute_database": "execute_database",
        "execute_web": "execute_web",
        "execute_hybrid": "execute_hybrid",
        "execute_forecast": "execute_forecast"
    })
    
    workflow.add_edge("execute_database", "store_memory")
    workflow.add_edge("execute_web", "store_memory")
    workflow.add_edge("execute_hybrid", "store_memory")
    workflow.add_edge("execute_forecast", "store_memory")
    
    workflow.add_edge("store_memory", END)
    
    return workflow.compile()

# Instantiate the compiled graph globally
agent_graph = build_graph()


# ─── Main Entrypoint ──────────────────────────────────────────────────────────

async def run_agent(
    question: str,
    session_id: str,
    dataframes: dict,
    clarification_choice: Optional[str] = None,
) -> AgentResponse:
    t0 = time.perf_counter()
    
    initial_state = {
        "question": question,
        "session_id": session_id,
        "dataframes": dataframes,
        "clarification_choice": clarification_choice,
        "execution_steps": []
    }
    
    try:
        final_state = await agent_graph.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"Agent graph error: {e}\n{traceback.format_exc()}")
        initial_state["error"] = str(e)
        initial_state["execution_steps"] = [{"step": "Agent error", "status": "error", "detail": str(e)}]
        final_state = initial_state

    final_state["total_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return AgentResponse(final_state)
