"""
Phase 16 — MCP Tools HTTP Endpoint
Exposes all agent tools as REST endpoints so external MCP-compatible
clients (or Gemini function-calling) can invoke them programmatically.

Endpoints:
  GET  /api/mcp/tools           → list all available tools + schemas
  POST /api/mcp/invoke          → invoke a tool by name
  GET  /api/mcp/health          → check Elasticsearch RAG store status
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

from app.tools.mcp_tools import MCP_TOOLS, MCPToolExecutor
from app.services.schema_intelligence import fetch_schema
from app.services.ingestion import load_session_data
from app.services.vector_store import es_health

router = APIRouter()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# List tools
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/tools")
async def list_tools():
    """Return all available MCP tool definitions with their JSON schemas."""
    return {
        "tools": MCP_TOOLS,
        "count": len(MCP_TOOLS),
        "description": "InsightFlow AI MCP Tool Registry",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Invoke a tool
# ─────────────────────────────────────────────────────────────────────────────

class InvokeRequest(BaseModel):
    session_id: str
    tool_name: str
    parameters: dict = {}


@router.post("/invoke")
async def invoke_tool(request: InvokeRequest):
    """
    Invoke an MCP tool by name with the given parameters.
    All tools require a valid session_id.
    """
    # Validate tool name
    valid_tools = {t["name"] for t in MCP_TOOLS}
    if request.tool_name not in valid_tools:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tool '{request.tool_name}'. Available: {sorted(valid_tools)}",
        )

    # Load session data
    schemas = await fetch_schema(request.session_id)
    if not schemas:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found.",
        )

    dataframes = {}
    for schema in schemas:
        tbl = schema["table_name"]
        try:
            dataframes[tbl] = load_session_data(request.session_id, tbl)
        except FileNotFoundError:
            logger.warning(f"Table {tbl} not found on disk for session {request.session_id}")

    # Execute
    executor = MCPToolExecutor(session_id=request.session_id, dataframes=dataframes)
    try:
        result = await executor.execute(request.tool_name, request.parameters)
        return {
            "tool": request.tool_name,
            "session_id": request.session_id,
            "result": result,
        }
    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# RAG / ES health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health")
async def mcp_health():
    """Check MCP subsystem health including Elasticsearch RAG store."""
    es_status = await es_health()
    return {
        "mcp_tools": len(MCP_TOOLS),
        "elasticsearch": es_status,
    }
