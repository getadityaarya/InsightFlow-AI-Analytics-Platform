"""
Database migrations — run once to initialise the SQL schema.
Usage: python database/migrate.py

For production, use Alembic:
  alembic init alembic
  alembic revision --autogenerate -m "init"
  alembic upgrade head
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.database import engine, Base
from app.models.session import Session, QueryLog, ToolCall


async def run_migrations():
    print("Running migrations...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ All tables created:")
    print("   - sessions")
    print("   - query_logs")
    print("   - tool_calls")
    print("\nDatabase ready at: insightflow.db")


if __name__ == "__main__":
    asyncio.run(run_migrations())
