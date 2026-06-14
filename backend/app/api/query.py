"""
Query API — Phases 5–17 entry point.
Runs the agent orchestrator and logs every query to the observability table.
"""

import time
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import run_agent
from app.services.schema_intelligence import fetch_schema
from app.services.ingestion import load_session_data
from app.services.memory import get_conversation_history, clear_session_memory
from app.services.observability import log_query
from app.core.database import get_db

router = APIRouter()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from upload response")
    question: str = Field(..., min_length=3, max_length=2000)
    clarification_choice: Optional[str] = Field(None)


class QueryResponse(BaseModel):
    question: str
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
    execution_steps: list[dict]
    total_time_ms: float


@router.post("/", response_model=QueryResponse)
async def run_query(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    schemas = await fetch_schema(request.session_id)
    if not schemas:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found. Please upload a dataset first.",
        )

    # Load all DataFrames for this session
    table_names = [s["table_name"] for s in schemas]
    dataframes = {}
    for tbl in table_names:
        try:
            dataframes[tbl] = load_session_data(request.session_id, tbl)
        except FileNotFoundError:
            logger.warning(f"Could not load table {tbl} for session {request.session_id}")

    if not dataframes:
        raise HTTPException(status_code=404, detail="Session data not found on disk. Please re-upload.")

    # Run agent
    agent_response = await run_agent(
        question=request.question,
        session_id=request.session_id,
        dataframes=dataframes,
        clarification_choice=request.clarification_choice,
    )

    result = agent_response.to_dict()

    # Phase 17: Log every query to SQL for observability
    await log_query(
        db=db,
        session_id=request.session_id,
        question=request.question,
        classification=result.get("classification"),
        sql_generated=result.get("sql"),
        sql_valid=result.get("sql_validation_error") is None,
        row_count=result.get("query_result", {}).get("row_count", 0) if result.get("query_result") else 0,
        chart_type=result.get("chart_type"),
        is_forecast=result.get("is_forecast", False),
        is_cached=result.get("query_result", {}).get("cached", False) if result.get("query_result") else False,
        total_time_ms=result.get("total_time_ms", 0),
        llm_calls=sum(1 for s in result.get("execution_steps", []) if "generated" in s.get("step", "").lower()),
        error=result.get("error"),
    )

    return QueryResponse(**result)


@router.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 20):
    history = await get_conversation_history(session_id, limit=limit)
    return {"session_id": session_id, "history": history, "count": len(history)}


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    deleted = await clear_session_memory(session_id)
    return {"session_id": session_id, "deleted_count": deleted}


@router.get("/schema/{session_id}")
async def get_schema(session_id: str):
    schemas = await fetch_schema(session_id)
    if not schemas:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "schemas": schemas}
