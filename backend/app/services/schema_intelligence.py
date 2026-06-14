"""
Phase 3+4 — Schema Intelligence & RAG Knowledge Base
Infers business meaning for every column, stores metadata in MongoDB,
and embeds it for vector search so the SQL agent never hallucinates column names.
"""

import pandas as pd
import json
import re
from typing import Any
import logging

from app.core.database import AsyncSessionLocal
from app.models.session import SchemaMetadata
from app.services.vector_store import index_schema, search_schema as vs_search_schema
from app.services.ingestion import _infer_column_type

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Schema Intelligence
# ─────────────────────────────────────────────────────────────────────────────

BUSINESS_MEANING_MAP = {
    # Revenue / Finance
    "revenue": "Total revenue generated",
    "amount": "Monetary amount",
    "price": "Unit price of product or service",
    "cost": "Cost incurred",
    "profit": "Net profit",
    "sales": "Sales value",
    "gmv": "Gross merchandise value",
    "spend": "Amount spent",
    "discount": "Discount applied",
    "tax": "Tax amount",

    # Time
    "date": "Date of the event",
    "created_at": "Record creation timestamp",
    "updated_at": "Last update timestamp",
    "order_date": "Date order was placed",
    "shipped_date": "Date item was shipped",
    "delivery_date": "Delivery date",

    # Identity
    "id": "Unique identifier",
    "customer_id": "Unique customer identifier",
    "order_id": "Unique order identifier",
    "product_id": "Unique product identifier",
    "user_id": "Unique user identifier",
    "session_id": "User session identifier",

    # People & Geography
    "customer": "Customer name or identifier",
    "name": "Entity name",
    "email": "Email address",
    "phone": "Phone number",
    "city": "City location",
    "state": "State or province",
    "country": "Country",
    "region": "Geographic region",
    "zip": "Postal code",
    "address": "Physical address",

    # Product
    "product": "Product name or category",
    "category": "Product or service category",
    "sku": "Stock keeping unit",
    "brand": "Brand name",
    "quantity": "Number of units",
    "units": "Units sold or ordered",
    "inventory": "Stock inventory level",
    "rating": "Customer rating",
    "review": "Customer review text",

    # Operations
    "status": "Current status of the record",
    "channel": "Sales or marketing channel",
    "source": "Traffic or data source",
    "campaign": "Marketing campaign",
    "segment": "Customer segment",
    "priority": "Priority level",
    "department": "Organizational department",
    "manager": "Manager name",
    "employee": "Employee name or ID",
}


def infer_schema_metadata(
    df: pd.DataFrame,
    table_name: str,
    session_id: str,
) -> dict:
    """
    Build a rich schema document for a table.
    Returns a structured dict ready for MongoDB storage.
    """
    columns = []
    for col in df.columns:
        col_type = _infer_column_type(df[col])
        description = _describe_column(col, df[col], col_type)
        sample_vals = _get_sample_values(df[col], col_type)

        columns.append({
            "name": col,
            "type": col_type,
            "description": description,
            "sample_values": sample_vals,
            "is_nullable": bool(df[col].isnull().any()),
            "cardinality": int(df[col].nunique()),
            "is_primary_key_candidate": _is_pk_candidate(df[col]),
            "is_foreign_key_candidate": _is_fk_candidate(col),
        })

    schema_doc = {
        "session_id": session_id,
        "table_name": table_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
        "rag_text": _build_rag_text(table_name, columns, len(df)),
    }
    return schema_doc


def _describe_column(col_name: str, series: pd.Series, col_type: str) -> str:
    """Infer a human-readable business description for a column."""
    name_lower = col_name.lower().replace("_", " ").replace("-", " ")

    # Direct match
    for keyword, desc in BUSINESS_MEANING_MAP.items():
        if keyword.replace("_", " ") in name_lower:
            return desc

    # Fallback by type
    type_fallbacks = {
        "currency": f"Monetary value — {col_name}",
        "numeric": f"Numeric measurement — {col_name}",
        "datetime": f"Date/time field — {col_name}",
        "category": f"Categorical grouping — {col_name}",
        "text": f"Text field — {col_name}",
    }
    return type_fallbacks.get(col_type, f"Data field — {col_name}")


def _get_sample_values(series: pd.Series, col_type: str, n: int = 5) -> list:
    """Pull representative sample values (never PII in real-world, simplified here)."""
    try:
        vals = series.dropna().unique()[:n]
        return [str(v) for v in vals]
    except Exception:
        return []


def _is_pk_candidate(series: pd.Series) -> bool:
    return series.nunique() == len(series) and series.notna().all()


def _is_fk_candidate(col_name: str) -> bool:
    return col_name.lower().endswith("_id") or col_name.lower().endswith("_key")


def _build_rag_text(table_name: str, columns: list[dict], row_count: int) -> str:
    """
    Build a searchable text document that the RAG retriever will embed.
    This is the bridge between natural language questions and the SQL engine.
    """
    lines = [
        f"Table: {table_name}",
        f"Description: This table contains {row_count} rows of business data.",
        "",
        "Columns:",
    ]
    for col in columns:
        sample_str = ", ".join(col["sample_values"][:3]) if col["sample_values"] else "N/A"
        lines.append(
            f"  - {col['name']} ({col['type']}): {col['description']}. "
            f"Example values: {sample_str}."
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Store schema in MongoDB
# ─────────────────────────────────────────────────────────────────────────────

async def store_schema(schema_doc: dict) -> str:
    """Upsert schema document into SQLite."""
    from sqlalchemy import select
    async with AsyncSessionLocal() as db:
        # Check if exists
        result = await db.execute(
            select(SchemaMetadata)
            .filter_by(session_id=schema_doc["session_id"], table_name=schema_doc["table_name"])
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            existing.rag_text = schema_doc.get("rag_text")
            existing.columns = schema_doc.get("columns", [])
            existing.row_count = schema_doc.get("row_count", 0)
        else:
            new_schema = SchemaMetadata(
                session_id=schema_doc["session_id"],
                table_name=schema_doc["table_name"],
                rag_text=schema_doc.get("rag_text"),
                columns=schema_doc.get("columns", []),
                row_count=schema_doc.get("row_count", 0),
            )
            db.add(new_schema)
        await db.commit()

    # Phase 4: Also index into vector store (Elasticsearch or in-memory fallback)
    await index_schema(
        session_id=schema_doc["session_id"],
        table_name=schema_doc["table_name"],
        rag_text=schema_doc.get("rag_text", ""),
        columns=schema_doc.get("columns", []),
    )
    logger.info(f"Schema stored + indexed: {schema_doc['table_name']}")
    return schema_doc["table_name"]


async def fetch_schema(session_id: str) -> list[dict]:
    """Retrieve all schemas for a session."""
    from sqlalchemy import select
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(SchemaMetadata)
            .filter_by(session_id=session_id)
        )
        schemas = result.scalars().all()
        return [
            {
                "session_id": s.session_id,
                "table_name": s.table_name,
                "rag_text": s.rag_text,
                "columns": s.columns,
                "row_count": s.row_count,
                "created_at": s.created_at,
            }
            for s in schemas
        ]


async def build_sql_context(session_id: str, query: str = "") -> str:
    """
    Assemble schema context for the SQL generation prompt.
    Phase 4 RAG: if a query is provided, retrieve only the most relevant
    table schemas from the vector store. Otherwise return all schemas.
    """
    if query:
        # RAG retrieval — fetch most relevant schema chunks
        results = await vs_search_schema(session_id, query, top_k=3)
        if results:
            parts = [r["rag_text"] for r in results]
            logger.info(f"RAG retrieved {len(parts)} schema chunk(s) for query: {query[:60]}")
            return "\n\n".join(parts)

    # Fallback: return full schema from MongoDB
    schemas = await fetch_schema(session_id)
    if not schemas:
        return "No schema available."
    return "\n\n".join(schema.get("rag_text", "") for schema in schemas)


def generate_create_table_sql(schema_doc: dict) -> str:
    """Generate a CREATE TABLE statement from schema metadata (for SQL context)."""
    cols = []
    for col in schema_doc["columns"]:
        sql_type = {
            "numeric": "REAL",
            "currency": "REAL",
            "datetime": "TEXT",
            "category": "TEXT",
            "text": "TEXT",
        }.get(col["type"], "TEXT")

        nullable = "" if col["is_nullable"] else " NOT NULL"
        cols.append(f"  [{col['name']}] {sql_type}{nullable}")

    col_str = ",\n".join(cols)
    return f"CREATE TABLE [{schema_doc['table_name']}] (\n{col_str}\n);"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — RAG-aware schema context retrieval
# ─────────────────────────────────────────────────────────────────────────────

async def build_sql_context_rag(session_id: str, question: str) -> str:
    """
    RAG-enhanced schema context: semantically retrieve the most relevant
    schema snippets for the question. Falls back to full schema if
    Elasticsearch is unavailable.
    """
    from app.services.rag_store import semantic_search

    hits = await semantic_search(question, session_id, top_k=3)
    if hits:
        logger.info(f"RAG retrieved {len(hits)} schema snippets for: {question[:60]}")
        return "\n\n".join(h["rag_text"] for h in hits)

    # Fallback: return full schema context
    return await build_sql_context(session_id)
