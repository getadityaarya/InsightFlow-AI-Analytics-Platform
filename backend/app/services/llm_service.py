"""
LLM Service — Gemini-powered SQL generation, insight narration, and question classification.
All prompts live here; the agent orchestrator calls these functions.
"""

import json
import re
import logging
from typing import Optional

try:
    import google.generativeai as genai
    _genai_available = True
except ImportError:
    genai = None  # type: ignore
    _genai_available = False

from app.core.config import settings
from app.prompts.templates import (
    SQL_GENERATION_PROMPT,
    BUSINESS_INSIGHT_PROMPT,
    QUESTION_CLASSIFIER_PROMPT,
    CLARIFICATION_PROMPT,
    FORECASTING_INSIGHT_PROMPT,
    WEB_SEARCH_SYNTHESIS_PROMPT,
)

logger = logging.getLogger(__name__)

# Configure Gemini
if _genai_available and settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    _model = genai.GenerativeModel(settings.GEMINI_MODEL)
else:
    _model = None
    if not _genai_available:
        logger.warning("google-generativeai not installed — using mock responses")
    else:
        logger.warning("GEMINI_API_KEY not set — LLM calls will use mock responses")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — SQL Generation
# ─────────────────────────────────────────────────────────────────────────────

async def generate_sql(
    question: str,
    schema_context: str,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Generate SQL from natural language question.
    Uses schema context (RAG) to ground the generation and prevent hallucination.
    """
    history_str = ""
    if conversation_history:
        history_str = "\n".join([
            f"User: {m['question']}\nSQL: {m.get('sql', '')}"
            for m in conversation_history[-3:]
        ])

    prompt = SQL_GENERATION_PROMPT.format(
        schema_context=schema_context,
        question=question,
        conversation_history=history_str or "None",
    )

    response = await _call_gemini(prompt)
    return _extract_sql(response)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 9 — Business Insight Engine
# ─────────────────────────────────────────────────────────────────────────────

async def generate_insight(
    question: str,
    sql: str,
    result_preview: str,
    schema_context: str,
) -> str:
    """
    Act as a senior business analyst to explain findings from query results.
    """
    prompt = BUSINESS_INSIGHT_PROMPT.format(
        question=question,
        sql=sql,
        result_preview=result_preview,
        schema_context=schema_context,
    )
    return await _call_gemini(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 10 — Question Classifier
# ─────────────────────────────────────────────────────────────────────────────

async def classify_question(question: str, schema_context: str) -> str:
    """
    Classify question as: DATABASE | WEB | HYBRID
    Prevents unnecessary web searches.
    """
    prompt = QUESTION_CLASSIFIER_PROMPT.format(
        question=question,
        schema_context=schema_context,
    )
    response = await _call_gemini(prompt)
    category = response.strip().upper()

    # Normalise
    for valid in ("DATABASE", "WEB", "HYBRID"):
        if valid in category:
            return valid
    return "DATABASE"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 14 — Clarification Agent
# ─────────────────────────────────────────────────────────────────────────────

async def check_ambiguity(question: str, schema_context: str) -> dict:
    """
    Detect ambiguous questions and return clarification options.
    Returns: {"is_ambiguous": bool, "options": [...], "rewritten": str}
    """
    prompt = CLARIFICATION_PROMPT.format(
        question=question,
        schema_context=schema_context,
    )
    response = await _call_gemini(prompt)

    try:
        data = _extract_json(response)
        return data
    except Exception:
        return {"is_ambiguous": False, "options": [], "rewritten": question}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 15 — Forecasting narrative
# ─────────────────────────────────────────────────────────────────────────────

async def narrate_forecast(
    question: str,
    forecast_summary: str,
    confidence_interval: str,
) -> str:
    prompt = FORECASTING_INSIGHT_PROMPT.format(
        question=question,
        forecast_summary=forecast_summary,
        confidence_interval=confidence_interval,
    )
    return await _call_gemini(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 11 — Web search synthesis
# ─────────────────────────────────────────────────────────────────────────────

async def synthesise_web_results(
    question: str,
    internal_findings: str,
    web_results: list[dict],
) -> str:
    web_text = "\n\n".join([
        f"Source: {r.get('url', 'unknown')}\n{r.get('content', '')[:1000]}"
        for r in web_results[:5]
    ])
    prompt = WEB_SEARCH_SYNTHESIS_PROMPT.format(
        question=question,
        internal_findings=internal_findings or "None",
        web_results=web_text,
    )
    return await _call_gemini(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _call_gemini(prompt: str) -> str:
    if _model is None:
        return _mock_response(prompt)

    try:
        response = await _model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise RuntimeError(f"LLM call failed: {e}") from e


def _extract_sql(response: str) -> str:
    """Pull SQL from markdown code blocks or raw text."""
    # Try ```sql ... ```
    match = re.search(r"```(?:sql)?\s*([\s\S]+?)```", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try to find SELECT ... ;
    match = re.search(r"(SELECT[\s\S]+?(?:;|$))", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()


def _extract_json(response: str) -> dict:
    """Pull JSON from LLM response (handles markdown fences)."""
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", response)
    if match:
        return json.loads(match.group(1))
    # Try bare JSON
    match = re.search(r"\{[\s\S]+\}", response)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No JSON found in response")


def _mock_response(prompt: str) -> str:
    """Mock responses when API key is not configured (dev mode)."""
    prompt_lower = prompt.lower()
    if "sql" in prompt_lower:
        return "SELECT * FROM main LIMIT 100;"
    if "classify" in prompt_lower:
        return "DATABASE"
    if "ambiguous" in prompt_lower:
        return json.dumps({"is_ambiguous": False, "options": [], "rewritten": ""})
    return "Mock insight: Analysis complete. Configure GEMINI_API_KEY for real insights."


# ─────────────────────────────────────────────────────────────────────────────
# Phase 13 — Agent Planner (LLM-powered step generation)
# ─────────────────────────────────────────────────────────────────────────────

async def generate_plan(
    question: str,
    schema_context: str,
    classification: str,
    is_forecast: bool,
) -> list[str]:
    """
    Use Gemini to generate a step-by-step execution plan for the question.
    Falls back to a static plan if the LLM fails or returns unparseable JSON.
    """
    from app.prompts.templates import AGENT_PLANNER_PROMPT

    prompt = AGENT_PLANNER_PROMPT.format(
        schema_context=schema_context,
        question=question,
    )
    try:
        response = await _call_gemini(prompt)
        steps_raw = _extract_json(response)
        if isinstance(steps_raw, list):
            return [s.get("description", str(s)) for s in steps_raw if isinstance(s, dict)]
    except Exception as e:
        logger.debug(f"LLM plan parsing failed ({e}) — using static plan")

    # Static fallback
    if is_forecast:
        return ["Detect date and value columns", "Run Prophet forecast",
                "Generate forecast chart", "Narrate findings"]
    plans = {
        "DATABASE": ["Retrieve schema context", "Generate SQL", "Validate SQL",
                     "Execute query", "Auto-select chart", "Generate business insight"],
        "WEB": ["Search external sources", "Retrieve top results", "Synthesise findings"],
        "HYBRID": ["Execute database query", "Search external sources",
                   "Synthesise internal and external findings"],
    }
    return plans.get(classification, plans["DATABASE"])


# ─────────────────────────────────────────────────────────────────────────────
# Phase 13 — Dynamic Agent Plan Generator
# ─────────────────────────────────────────────────────────────────────────────

async def generate_agent_plan(
    question: str,
    schema_context: str,
    classification: str,
    is_forecast: bool,
) -> list[str]:
    """
    Generate a dynamic LLM-driven execution plan for the agent.
    Falls back to static plan if LLM is unavailable.
    """
    from app.prompts.templates import AGENT_PLANNER_PROMPT
    prompt = AGENT_PLANNER_PROMPT.format(
        schema_context=schema_context,
        question=question,
        classification=classification,
        is_forecast=str(is_forecast),
    )
    try:
        response = await _call_gemini(prompt)
        parsed = _extract_json(response)
        if isinstance(parsed, list):
            return [step.get("description", str(step)) for step in parsed if step]
    except Exception as e:
        logger.warning(f"Dynamic plan failed ({e}), using static plan")

    # Static fallback plans
    if is_forecast:
        return ["Detect date and value columns", "Run Prophet forecast", "Build forecast chart", "Narrate findings"]
    plans = {
        "DATABASE": ["Retrieve schema via RAG", "Generate SQL", "Validate SQL", "Execute query", "Build chart", "Generate insight"],
        "WEB":      ["Search external sources", "Retrieve top results", "Synthesise findings"],
        "HYBRID":   ["Query database", "Search external sources", "Synthesise internal + external findings"],
    }
    return plans.get(classification, plans["DATABASE"])
