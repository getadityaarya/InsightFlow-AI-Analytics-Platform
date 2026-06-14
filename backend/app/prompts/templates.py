"""
Prompt Templates — All LLM prompts centralised here for easy iteration.
Each prompt is carefully engineered to reduce hallucination and improve output quality.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — SQL Generation
# ─────────────────────────────────────────────────────────────────────────────

SQL_GENERATION_PROMPT = """\
You are an expert SQL analyst. Generate a SQL query to answer the user's business question.

STRICT RULES:
1. Use ONLY the column names and tables listed in the schema below — never invent columns
2. Return ONLY the SQL query, nothing else (no explanation, no markdown)
3. Use SQLite-compatible syntax
4. For text comparisons, always use LOWER() for case-insensitive matching
5. Always include appropriate WHERE clauses, LIMIT clauses (max 10000 rows)
6. For aggregations, always include meaningful aliases (e.g. SUM(amount) AS total_revenue)
7. Handle NULLs appropriately with COALESCE where needed

DATABASE SCHEMA:
{schema_context}

CONVERSATION HISTORY (for context):
{conversation_history}

USER QUESTION: {question}

SQL QUERY:
"""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 9 — Business Insight Engine
# ─────────────────────────────────────────────────────────────────────────────

BUSINESS_INSIGHT_PROMPT = """\
You are a senior business analyst with 15 years of experience.
Analyse the following query result and provide actionable business insights.

GUIDELINES:
- Write in clear, professional business language
- Highlight the most important trend or finding first
- Quantify everything (e.g. "12% increase", "top 3 regions contribute 67%")
- Suggest 1-2 concrete next steps or questions to investigate
- Keep response to 3-5 bullet points
- If the result is empty, explain what that means for the business

USER QUESTION: {question}

SQL USED:
{sql}

QUERY RESULT PREVIEW (first 20 rows):
{result_preview}

SCHEMA CONTEXT:
{schema_context}

BUSINESS INSIGHTS:
"""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 10 — Question Classifier
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_CLASSIFIER_PROMPT = """\
Classify this user question into exactly one of: DATABASE, WEB, or HYBRID.

DATABASE: Answer requires only querying the uploaded dataset
  Examples: "top 5 products by revenue", "show sales by region", "which customers churned"

WEB: Answer requires current/external information not in the dataset
  Examples: "current inflation rate", "what is the market cap of Apple", "latest industry trends"

HYBRID: Requires both dataset query AND external context to fully answer
  Examples: "why are EV sales increasing in our data?", "how does our performance compare to industry benchmarks?",
            "what caused the spike in Q2?" (could need external news/events)

AVAILABLE SCHEMA:
{schema_context}

USER QUESTION: {question}

Classification (respond with exactly one word — DATABASE, WEB, or HYBRID):
"""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 14 — Clarification Agent
# ─────────────────────────────────────────────────────────────────────────────

CLARIFICATION_PROMPT = """\
Analyse whether this user question is ambiguous given the available schema.

If ambiguous, provide 2-4 clear interpretation options.
If clear, just confirm it's unambiguous.

Return a JSON object in this exact format:
{{
  "is_ambiguous": true/false,
  "ambiguity_reason": "brief explanation if ambiguous, else null",
  "options": [
    {{"label": "Option description", "rewritten_question": "Specific unambiguous question"}},
    ...
  ],
  "rewritten": "Best guess at user intent (for unambiguous questions)"
}}

SCHEMA:
{schema_context}

QUESTION: {question}

JSON RESPONSE:
"""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 15 — Forecasting narrative
# ─────────────────────────────────────────────────────────────────────────────

FORECASTING_INSIGHT_PROMPT = """\
You are a data scientist explaining forecast results to a business audience.

Interpret the following forecast in plain business language:
- What is the expected trend?
- How confident is the model?
- What should the business do with this information?
- What factors could affect the forecast accuracy?

USER QUESTION: {question}

FORECAST SUMMARY:
{forecast_summary}

CONFIDENCE INTERVALS:
{confidence_interval}

Keep response to 4-5 sentences, business-focused, avoid statistical jargon.

FORECAST NARRATIVE:
"""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 11 — Web search synthesis
# ─────────────────────────────────────────────────────────────────────────────

WEB_SEARCH_SYNTHESIS_PROMPT = """\
You are an analyst synthesising internal data findings with external web research.
Keep internal and external evidence clearly separated in your response.

IMPORTANT:
- Never mix internal findings with external ones
- Clearly label the source of each insight
- If internal and external data conflict, note the discrepancy
- Be concise — 3-5 bullet points max per section

USER QUESTION: {question}

INTERNAL FINDINGS (from uploaded dataset):
{internal_findings}

EXTERNAL WEB RESULTS:
{web_results}

RESPONSE FORMAT:

**Internal Findings (from your data):**
• [bullet points from dataset analysis]

**External Context (from web search):**
• [bullet points from web research]

**Combined Interpretation:**
[1-2 sentences synthesising both sources]
"""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 13 — Agent Planner
# ─────────────────────────────────────────────────────────────────────────────

AGENT_PLANNER_PROMPT = """\
You are an intelligent data analyst agent. Create a step-by-step execution plan for this question.

Available tools:
- query_database(sql): Execute SQL against the uploaded dataset
- search_web(query): Search for external information
- generate_chart(data, chart_type): Create a visualisation
- run_forecast(column, periods): Run time-series forecast
- get_conversation_history(): Retrieve past questions/answers

Rules:
- Only include steps that are necessary
- Be specific about what each step should do
- Keep plan to 3-6 steps maximum

SCHEMA:
{schema_context}

QUESTION: {question}

Return a JSON array of steps:
[
  {{"step": 1, "tool": "tool_name", "description": "what this step does", "input": "specific input"}},
  ...
]

EXECUTION PLAN:
"""
