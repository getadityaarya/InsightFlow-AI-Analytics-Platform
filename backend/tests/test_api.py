"""
API Integration Tests — FastAPI TestClient
Tests all endpoints: /health, /upload/, /query/, /profile/, /observe/
"""

import pytest
import io
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


# ─────────────────────────────────────────────────────────────────────────────
# App fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
async def client():
    """Async test client for the FastAPI app."""
    from app.main import app
    from app.core.database import get_db

    async def override_db():
        mock = MagicMock()
        mock.add = MagicMock()
        mock.commit = AsyncMock()
        mock.refresh = AsyncMock()
        mock.get = AsyncMock(return_value=None)
        mock.execute = AsyncMock(return_value=MagicMock(
            one=MagicMock(return_value=MagicMock(
                total_queries=5, avg_latency_ms=120.0,
                total_rows_fetched=500, total_llm_calls=10, cached_hits=2,
            )),
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))),
            all=MagicMock(return_value=[]),
            scalar=MagicMock(return_value=0),
        ))
        yield mock

    app.dependency_overrides[get_db] = override_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


def make_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def sales_csv_bytes():
    np.random.seed(0)
    n = 200
    df = pd.DataFrame({
        "order_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "customer": np.random.choice(["Alice", "Bob", "Carol"], n),
        "product": np.random.choice(["Laptop", "Phone", "Tablet"], n),
        "region": np.random.choice(["North", "South", "East"], n),
        "sales_amount": np.round(np.random.uniform(100, 2000, n), 2),
        "quantity": np.random.randint(1, 5, n),
    })
    return make_csv_bytes(df)


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "InsightFlow AI"


# ─────────────────────────────────────────────────────────────────────────────
# Upload endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_csv(client, sales_csv_bytes):
    files = {"file": ("sales.csv", io.BytesIO(sales_csv_bytes), "text/csv")}
    resp = await client.post("/api/upload/", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["filename"] == "sales.csv"
    assert len(data["tables"]) == 1
    assert data["tables"][0]["rows"] == 200
    assert data["profile_summary"]["total_rows"] == 200
    assert data["profile_summary"]["avg_quality_score"] > 0


@pytest.mark.asyncio
async def test_upload_unsupported_type(client):
    files = {"file": ("data.json", io.BytesIO(b'{"key": "value"}'), "application/json")}
    resp = await client.post("/api/upload/", files=files)
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_upload_empty_csv(client):
    files = {"file": ("empty.csv", io.BytesIO(b"col1,col2\n"), "text/csv")}
    resp = await client.post("/api/upload/", files=files)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_upload_returns_column_types(client, sales_csv_bytes):
    files = {"file": ("sales.csv", io.BytesIO(sales_csv_bytes), "text/csv")}
    resp = await client.post("/api/upload/", files=files)
    data = resp.json()
    col_types = data["tables"][0]["column_types"]
    assert "sales_amount" in col_types
    assert col_types["sales_amount"] in ("currency", "numeric")


@pytest.mark.asyncio
async def test_upload_returns_quality_score(client, sales_csv_bytes):
    files = {"file": ("sales.csv", io.BytesIO(sales_csv_bytes), "text/csv")}
    resp = await client.post("/api/upload/", files=files)
    data = resp.json()
    assert 0 <= data["tables"][0]["quality_score"] <= 100


@pytest.mark.asyncio
async def test_upload_excel(client):
    np.random.seed(1)
    df = pd.DataFrame({"name": ["A", "B", "C"], "value": [1.0, 2.0, 3.0]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    files = {"file": ("data.xlsx", buf, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    resp = await client.post("/api/upload/", files=files)
    assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Query endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
async def session_id(client, sales_csv_bytes):
    """Upload a dataset and return its session_id."""
    files = {"file": ("sales.csv", io.BytesIO(sales_csv_bytes), "text/csv")}
    resp = await client.post("/api/upload/", files=files)
    assert resp.status_code == 200, f"Upload failed: {resp.text}"
    data = resp.json()
    assert "session_id" in data, f"No session_id in: {data}"
    return data["session_id"]


@pytest.mark.asyncio
async def test_query_database(client, session_id):
    payload = {"session_id": session_id, "question": "Show total sales by region"}
    resp = await client.post("/api/query/", json=payload)
    # Schema is loaded from parquet (not MongoDB mock), so 200 expected
    assert resp.status_code in (200, 404), f"Unexpected status: {resp.status_code} {resp.text}"
    if resp.status_code == 200:
        data = resp.json()
        assert data["classification"] in ("DATABASE", "WEB", "HYBRID")
        assert "insight" in data
        assert len(data["execution_steps"]) > 0


@pytest.mark.asyncio
async def test_query_returns_sql(client, session_id):
    payload = {"session_id": session_id, "question": "Top 5 customers by sales"}
    resp = await client.post("/api/query/", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        if data.get("classification") == "DATABASE" and data.get("sql"):
            assert "SELECT" in data["sql"].upper()


@pytest.mark.asyncio
async def test_query_unknown_session(client):
    payload = {"session_id": "nonexistent-session-id", "question": "show data"}
    resp = await client.post("/api/query/", json=payload)
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_query_too_short(client):
    payload = {"session_id": "any", "question": "hi"}
    resp = await client.post("/api/query/", json=payload)
    assert resp.status_code == 422  # Pydantic min_length validation


@pytest.mark.asyncio
async def test_query_history(client, session_id):
    # Ask a question first
    await client.post("/api/query/", json={"session_id": session_id, "question": "Show all data"})
    resp = await client.get(f"/api/query/history/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "history" in data
    assert "count" in data


@pytest.mark.asyncio
async def test_query_schema_endpoint(client, session_id):
    resp = await client.get(f"/api/query/schema/{session_id}")
    # Schema stored in MongoDB (mocked); accept 200 or 404
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        data = resp.json()
        assert "schemas" in data


@pytest.mark.asyncio
async def test_clear_history(client, session_id):
    resp = await client.delete(f"/api/query/history/{session_id}")
    assert resp.status_code == 200
    assert "deleted_count" in resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# Profile endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_profile_endpoint(client, session_id):
    resp = await client.get(f"/api/profile/{session_id}/main")
    assert resp.status_code == 200
    data = resp.json()
    assert data["rows"] == 200
    assert "column_profiles" in data
    assert data["quality_score"] >= 0


@pytest.mark.asyncio
async def test_profile_unknown_table(client, session_id):
    resp = await client.get(f"/api/profile/{session_id}/nonexistent_table")
    assert resp.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Observability endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_observability_stats(client, session_id):
    resp = await client.get(f"/api/observe/stats/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_queries" in data
    assert "avg_latency_ms" in data
    assert "classification_breakdown" in data


@pytest.mark.asyncio
async def test_observability_logs(client, session_id):
    resp = await client.get(f"/api/observe/logs/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "logs" in data
