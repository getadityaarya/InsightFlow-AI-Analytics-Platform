"""
import os
os.environ["DEBUG"] = "true"  # Disables rate limiting during tests

conftest.py — shared pytest fixtures and test configuration.
Mocks MongoDB and SQLAlchemy so tests run without any external services.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


# ── Event loop ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Mock MongoDB ──────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_mongo(monkeypatch):
    """
    Auto-used fixture: patches all MongoDB calls so tests never need
    a real MongoDB connection.
    """
    mock_collection = MagicMock()

    # replace_one, insert_one, find, find_one, count_documents, delete_many
    mock_collection.replace_one = AsyncMock(return_value=MagicMock(upserted_id="mock-id"))
    mock_collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id="mock-id"))
    mock_collection.find_one = AsyncMock(return_value=None)
    mock_collection.delete_many = AsyncMock(return_value=MagicMock(deleted_count=0))
    mock_collection.count_documents = AsyncMock(return_value=0)

    # find(...).sort(...).limit(...).to_list(...)
    mock_cursor = MagicMock()
    mock_cursor.sort = MagicMock(return_value=mock_cursor)
    mock_cursor.limit = MagicMock(return_value=mock_cursor)
    mock_cursor.to_list = AsyncMock(return_value=[])
    mock_collection.find = MagicMock(return_value=mock_cursor)

    monkeypatch.setattr(
        "app.core.database.get_schema_collection",
        lambda: mock_collection,
    )
    monkeypatch.setattr(
        "app.core.database.get_memory_collection",
        lambda: mock_collection,
    )
    monkeypatch.setattr(
        "app.core.database.get_sessions_collection",
        lambda: mock_collection,
    )
    return mock_collection


# ── Mock Gemini LLM ───────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    """Patches Gemini so no real API calls are made during tests."""
    async def mock_call_gemini(prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "sql" in prompt_lower and "select" not in prompt_lower:
            return "SELECT * FROM main LIMIT 10;"
        if "classify" in prompt_lower:
            return "DATABASE"
        if "ambiguous" in prompt_lower:
            import json
            return json.dumps({"is_ambiguous": False, "options": [], "rewritten": "test question"})
        if "insight" in prompt_lower or "analyst" in prompt_lower:
            return "• Revenue is performing well.\n• Top region is North with 32% share.\n• Recommend investigating Q3 dip."
        if "forecast" in prompt_lower:
            return "Expected moderate growth of 8% over the forecast period based on historical seasonality."
        return "Mock LLM response."

    monkeypatch.setattr("app.services.llm_service._call_gemini", mock_call_gemini)
    return mock_call_gemini


# ── Mock web search ───────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_web_search(monkeypatch):
    """Patches Tavily web search."""
    async def mock_search(query: str, max_results: int = 5):
        return [
            {"title": "Mock Result", "url": "https://example.com", "content": "Mock web content.", "score": 0.9}
        ]
    monkeypatch.setattr("app.services.web_search.search_web", mock_search)
    return mock_search


# ── Temp upload dir ───────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def temp_upload_dir(tmp_path, monkeypatch):
    """Redirect uploads to a temp directory for each test."""
    monkeypatch.setattr("app.core.config.settings.UPLOAD_DIR", str(tmp_path))
    import os
    os.makedirs(str(tmp_path), exist_ok=True)
    return tmp_path


# ── Mock SQLAlchemy AsyncSession (get_db) ─────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_db(monkeypatch):
    """
    Patches FastAPI's get_db dependency so API tests never need a real DB.
    """
    mock_session = MagicMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_session.get = AsyncMock(return_value=None)
    mock_session.execute = AsyncMock(return_value=MagicMock(
        one=MagicMock(return_value=MagicMock(
            total_queries=0, avg_latency_ms=0.0,
            total_rows_fetched=0, total_llm_calls=0, cached_hits=0
        )),
        scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))),
        all=MagicMock(return_value=[]),
        scalar=MagicMock(return_value=0),
    ))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    async def override_get_db():
        yield mock_session

    monkeypatch.setattr("app.core.database.AsyncSessionLocal", MagicMock(return_value=mock_session))
    return mock_session


# ── Mock vector store ─────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_vector_store(monkeypatch):
    """Patches vector store so no Gemini API needed for embeddings."""
    monkeypatch.setattr("app.services.vector_store._get_embedding", lambda text, task_type="": [0.1] * 768)
    return None


# ── Reset rate limiter between tests ─────────────────────────────────────────
@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Clear in-process rate limit counters between tests."""
    try:
        from app.main import _rate_store
        _rate_store.clear()
    except ImportError:
        pass
    yield
    try:
        from app.main import _rate_store
        _rate_store.clear()
    except ImportError:
        pass
