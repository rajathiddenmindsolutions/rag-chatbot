"""Serper.dev Google Search Tool for Web Retrieval."""

import httpx
import structlog
from langchain_core.tools import tool
from ragchat.config import settings

logger = structlog.get_logger(__name__)

SERPER_API_URL = "https://google.serper.dev/search"


@tool
async def google_search_tool(query: str) -> str:
    """Use this tool to search Google for real-time information, current world facts, news, people, political leaders (e.g. Chief Ministers, Presidents), or topics that are NOT covered in the uploaded local documents.

    Args:
        query: The search query string to look up on Google.

    Returns:
        A formatted string of Google search result titles, snippets, and source URLs.
    """
    api_key = settings.serper_api_key
    if not api_key:
        logger.warning("serper_api_key_not_configured")
        return "Google search is unavailable because SERPER_API_KEY is not configured in .env."

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": 5,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(SERPER_API_URL, headers=headers, json=payload)
            if response.status_code != 200:
                logger.error("serper_api_error", status_code=response.status_code, body=response.text)
                return f"Google search failed with status code {response.status_code}."

            data = response.json()
            organic_results = data.get("organic", [])

            if not organic_results:
                return "No search results found on Google."

            formatted = []
            for item in organic_results[:5]:
                formatted.append(f"Title: {item.get('title', '')}\nSnippet: {item.get('snippet', '')}\nURL: {item.get('link', '')}")

            logger.info("serper_google_search_success", query=query, count=len(formatted))
            return "\n\n".join(formatted)

    except Exception as exc:
        logger.error("serper_search_failed", error=str(exc))
        return f"Google search encountered an error: {exc}"


async def search_google_serper(query: str, num_results: int = 5) -> list[dict]:
    """Direct helper to execute Serper search and return structured result dicts."""
    api_key = settings.serper_api_key
    if not api_key:
        return []

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": num_results,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(SERPER_API_URL, headers=headers, json=payload)
            if response.status_code != 200:
                return []

            data = response.json()
            organic_results = data.get("organic", [])

            search_results = []
            for item in organic_results:
                search_results.append({
                    "title": item.get("title", ""),
                    "text": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "section_path": "web_search",
                    "document_id": item.get("link", ""),
                    "chunk_index": 0,
                    "chunking_strategy": "web_search",
                    "score": 1.0,
                })

            return search_results
    except Exception:
        return []
