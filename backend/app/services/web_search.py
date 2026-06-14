"""
Phase 11 — Web Search Module
Only triggered for WEB or HYBRID question classifications.
Uses Tavily AI for structured web search results.
"""

import logging
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


async def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web for external context.
    Returns list of {title, url, content, score} dicts.
    """
    if not settings.TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set — returning mock web results")
        return _mock_results(query)

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        results = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
        )
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
            }
            for r in results.get("results", [])
        ]

    except ImportError:
        logger.warning("tavily-python not installed. Run: pip install tavily-python")
        return _mock_results(query)

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []


def _mock_results(query: str) -> list[dict]:
    """Mock search results for development without API key."""
    return [
        {
            "title": f"Market Analysis: {query}",
            "url": "https://example.com/analysis",
            "content": f"Industry experts note that {query} has seen significant changes. "
                       f"Recent reports indicate market conditions are evolving due to "
                       f"macroeconomic factors, consumer behaviour shifts, and technological adoption.",
            "score": 0.85,
        },
        {
            "title": f"Industry Report: Trends in {query}",
            "url": "https://example.com/trends",
            "content": f"The latest industry data suggests {query} is influenced by seasonal patterns "
                       f"and competitive dynamics. Analysts recommend monitoring key metrics closely.",
            "score": 0.72,
        },
    ]
