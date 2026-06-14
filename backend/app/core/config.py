"""
Core configuration — environment-driven settings with sane defaults.
Uses Pydantic v2 model_config style.
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List
import os


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # App
    APP_NAME: str = "InsightFlow AI"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"

    # API Keys
    GEMINI_API_KEY: str = ""
    TAVILY_API_KEY: str = ""

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:80"]

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./insightflow.db"


    # File storage
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: List[str] = ["csv", "xlsx", "xls", "sqlite", "db"]

    # LLM
    GEMINI_MODEL: str = "gemini-1.5-pro"
    MAX_SQL_ROWS: int = 10_000
    MAX_CHART_ROWS: int = 500

    # Observability
    DYNATRACE_URL: str = ""
    DYNATRACE_TOKEN: str = ""


settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
