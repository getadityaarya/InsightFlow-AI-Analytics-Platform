"""
Phase 4 — RAG Knowledge Base wrapper
Redirects to vector_store.py which now uses FAISS.
"""

from app.services.vector_store import (
    index_schema as vs_index_schema,
    search_schema,
    delete_session_index,
    es_health
)

async def index_schema(schema_doc: dict) -> bool:
    await vs_index_schema(
        session_id=schema_doc["session_id"],
        table_name=schema_doc["table_name"],
        rag_text=schema_doc.get("rag_text", ""),
        columns=schema_doc.get("columns", [])
    )
    return True

async def ensure_index() -> bool:
    return True

async def semantic_search(question: str, session_id: str, top_k: int = 3) -> list[dict]:
    return await search_schema(session_id, question, top_k)

def _hash_embedding(text: str, dims: int = 384) -> list[float]:
    import hashlib, struct
    h = hashlib.sha256(text.encode()).digest()
    extended = (h * ((dims * 4 // len(h)) + 1))[: dims * 4]
    raw = struct.unpack(f"{dims}f", extended)
    norm = max(abs(v) for v in raw) or 1.0
    return [v / norm for v in raw]

