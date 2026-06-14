"""
Phase 4 — RAG Knowledge Base via FAISS & Gemini Embeddings
Embeds schema metadata as searchable documents so the SQL agent can
retrieve only the relevant columns/tables for a given question.
"""

import logging
import faiss
import numpy as np
import google.generativeai as genai
from app.core.config import settings

logger = logging.getLogger(__name__)

# ── FAISS Store (per session) ──────────────────────────────────────────────────
# Map of session_id -> (faiss_index, list_of_metadata_dicts)
_session_stores: dict[str, tuple[faiss.IndexFlatL2, list[dict]]] = {}

# Embedding dimension for text-embedding-004 is 768
EMBEDDING_DIM = 768


def _get_embedding(text: str, task_type: str = "retrieval_document") -> list[float]:
    """Get Gemini embedding for a text."""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type=task_type,
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Gemini embedding error: {e}")
        # Try fallback to older model if 004 is not available
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
            )
            return result['embedding']
        except Exception as e2:
            logger.error(f"Gemini embedding fallback error: {e2}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def index_schema(session_id: str, table_name: str, rag_text: str, columns: list[dict]):
    """
    Index a table's RAG text into FAISS using Gemini embeddings.
    """
    doc = {
        "session_id": session_id,
        "table_name": table_name,
        "rag_text": rag_text,
        "column_names": [c["name"] for c in columns],
        "column_types": {c["name"]: c["type"] for c in columns},
        "column_descriptions": {c["name"]: c.get("description", "") for c in columns},
    }

    embedding = _get_embedding(rag_text, task_type="retrieval_document")
    if embedding:
        doc["embedding"] = embedding
    else:
        logger.warning(f"Failed to embed {table_name}, will fallback to keyword search")
        doc["embedding"] = [0.0] * EMBEDDING_DIM # Dummy embedding

    if session_id not in _session_stores:
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        _session_stores[session_id] = (index, [])

    index, docs = _session_stores[session_id]

    # Check if table already exists, remove it if we had a way, but FAISS FlatL2 doesn't support deletion.
    # So we rebuild the index for the session if we are overwriting a table.
    existing_idx = -1
    for i, d in enumerate(docs):
        if d["table_name"] == table_name:
            existing_idx = i
            break
            
    if existing_idx >= 0:
        docs[existing_idx] = doc
        # Rebuild index
        new_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        vectors = np.array([d["embedding"] for d in docs], dtype=np.float32)
        if len(vectors) > 0:
            new_index.add(vectors)
        _session_stores[session_id] = (new_index, docs)
    else:
        docs.append(doc)
        vector = np.array([doc["embedding"]], dtype=np.float32)
        index.add(vector)

    logger.info(f"Schema indexed in FAISS: session={session_id}, table={table_name}")


async def search_schema(session_id: str, query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve the most relevant schema chunks for a user query via semantic search.
    """
    if session_id not in _session_stores:
        return []

    index, docs = _session_stores[session_id]
    if not docs:
        return []

    query_embedding = _get_embedding(query, task_type="retrieval_query")
    
    if query_embedding:
        # Perform FAISS search
        q_vector = np.array([query_embedding], dtype=np.float32)
        k = min(top_k, len(docs))
        distances, indices = index.search(q_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(docs):
                d = docs[idx]
                # Convert L2 distance to a similarity score (lower distance = higher similarity)
                # Ensure we don't return dummy embeddings as high matches
                if sum(d["embedding"]) == 0.0:
                    continue
                score = 1.0 / (1.0 + distances[0][i])
                results.append({
                    "table_name": d["table_name"],
                    "rag_text": d["rag_text"],
                    "score": float(score),
                    "column_names": d.get("column_names", []),
                })
        
        if results:
            return results

    # Fallback to keyword search if embedding failed or all results were dummy
    logger.info("Falling back to keyword search")
    return _search_memory_keyword(docs, query, top_k)


async def delete_session_index(session_id: str):
    """Remove all indexed documents for a session (cleanup)."""
    _session_stores.pop(session_id, None)


def _search_memory_keyword(docs: list[dict], query: str, top_k: int) -> list[dict]:
    """
    Simple keyword overlap search — fallback if Gemini API fails.
    """
    query_tokens = set(query.lower().split())
    scored = []
    for doc in docs:
        text = doc.get("rag_text", "").lower()
        score = sum(1 for tok in query_tokens if tok in text)
        col_names = [c.lower() for c in doc.get("column_names", [])]
        score += sum(2 for tok in query_tokens if any(tok in col for col in col_names))
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "table_name": d["table_name"],
            "rag_text": d["rag_text"],
            "score": float(s),
            "column_names": d.get("column_names", []),
        }
        for s, d in scored[:top_k]
        if s > 0
    ] or [
        {"table_name": d["table_name"], "rag_text": d["rag_text"], "score": 0.0, "column_names": d.get("column_names", [])}
        for d in docs[:top_k]
    ]


async def es_health() -> dict:
    """Return FAISS health status (reusing old ES endpoint name)."""
    return {
        "status": "connected",
        "version": faiss.__version__,
        "index": "faiss_memory",
        "mode": "faiss_local",
    }
