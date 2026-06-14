# Contributing to InsightFlow AI

## Project structure recap

```
backend/app/
├── api/          ← HTTP endpoints (one file per domain)
├── agents/       ← Orchestrator — the brain of the agent
├── core/         ← Config, database connections
├── models/       ← SQLAlchemy ORM models
├── prompts/      ← All LLM prompt templates (edit here to tune behaviour)
├── services/     ← Business logic (no HTTP here)
└── tools/        ← MCP tool definitions + executor
```

## Adding a new LLM prompt

1. Add the template string to `app/prompts/templates.py`
2. Add a corresponding async function in `app/services/llm_service.py`
3. Call it from the orchestrator or relevant service
4. Add a test in `tests/test_extended.py` using the mocked `_call_gemini`

## Adding a new API endpoint

1. Create or extend a file in `app/api/`
2. Register the router in `app/main.py`
3. Add integration tests

## Adding a new MCP tool

1. Add the tool schema to `MCP_TOOLS` list in `app/tools/mcp_tools.py`
2. Add the handler method to `MCPToolExecutor.execute()`
3. Add a test in `TestMCPTools`

## Code style

- **Async everywhere** — all DB calls, LLM calls, and I/O must be `async def`
- **No blocking calls in endpoints** — use `asyncio.to_thread()` for CPU-heavy work
- **Type hints** — all function signatures must have type hints
- **Docstrings** — all public functions need a one-line docstring minimum

## Running the test suite

```bash
cd backend
pytest tests/ -v -m "not slow"   # fast tests only (CI)
pytest tests/ -v                  # all tests including Prophet
```

All PRs must pass `pytest tests/ -m "not slow"` with zero failures.

## Environment for local dev

Minimum `.env` to run tests:

```env
GEMINI_API_KEY=dummy        # mocked in tests
MONGODB_URL=mongodb://localhost:27017   # or leave unset — mocked in conftest
```

The `conftest.py` auto-mocks MongoDB, Gemini, Tavily, and the upload
directory — tests never hit real external services.
