"""
SQLAlchemy ORM Models
Lightweight — MongoDB handles the heavy document storage.
SQL tables track session metadata and observability events.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON
from sqlalchemy.sql import func
from app.core.database import Base


class Session(Base):
    """Tracks active upload sessions."""
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)           # uuid
    filename = Column(String, nullable=False)
    table_names = Column(JSON, default=list)         # list of table names
    total_rows = Column(Integer, default=0)
    total_columns = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)
    fingerprint = Column(String, nullable=True)      # SHA-256 for dedup
    created_at = Column(DateTime, server_default=func.now())
    last_active = Column(DateTime, onupdate=func.now())


class QueryLog(Base):
    """Observability — every query execution logged here (Phase 17)."""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    question = Column(Text, nullable=False)
    classification = Column(String, nullable=True)   # DATABASE / WEB / HYBRID
    sql_generated = Column(Text, nullable=True)
    sql_valid = Column(Integer, default=1)           # 0 = blocked
    row_count = Column(Integer, default=0)
    chart_type = Column(String, nullable=True)
    is_forecast = Column(Integer, default=0)
    is_cached = Column(Integer, default=0)
    total_time_ms = Column(Float, default=0.0)
    sql_time_ms = Column(Float, default=0.0)
    llm_calls = Column(Integer, default=0)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class ToolCall(Base):
    """Tracks individual tool invocations for the agent planner (Phase 17)."""
    __tablename__ = "tool_calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_log_id = Column(Integer, nullable=False, index=True)
    tool_name = Column(String, nullable=False)
    input_summary = Column(Text, nullable=True)
    latency_ms = Column(Float, default=0.0)
    success = Column(Integer, default=1)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class Memory(Base):
    """Stores conversational memory instead of MongoDB."""
    __tablename__ = "memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    question = Column(Text, nullable=False)
    classification = Column(String, nullable=True)
    sql = Column(Text, nullable=True)
    result_preview = Column(Text, nullable=True)
    insight = Column(Text, nullable=True)
    chart_type = Column(String, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())


class SchemaMetadata(Base):
    """Stores schema metadata instead of MongoDB."""
    __tablename__ = "schemas"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    table_name = Column(String, nullable=False)
    rag_text = Column(Text, nullable=True)
    columns = Column(JSON, default=list)
    row_count = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
