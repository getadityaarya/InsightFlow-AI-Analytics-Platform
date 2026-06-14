"""Profile API router."""

from fastapi import APIRouter, HTTPException
from app.services.ingestion import load_session_data, profile_dataframe

router = APIRouter()


@router.get("/{session_id}/{table_name}")
async def get_table_profile(session_id: str, table_name: str):
    """Get detailed data quality profile for a specific table."""
    try:
        df = load_session_data(session_id, table_name)
        profile = profile_dataframe(df, table_name)
        return profile
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Table not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
