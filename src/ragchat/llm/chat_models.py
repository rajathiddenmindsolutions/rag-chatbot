"""LLM Provider factory module with client caching and strict provider isolation."""

import os
from functools import lru_cache
import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from ragchat.config import settings

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=16)
def _cached_chat_model(selected_provider: str, target_model: str, temperature: float) -> BaseChatModel:
    """Internal LRU-cached constructor to reuse HTTP connections and LLM client instances."""
    if selected_provider in ("gemini", "google"):
        api_key = (
            settings.google_api_key
            or settings.gemni_api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMNI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        logger.info("instantiating_cached_gemini_client", model=target_model, temperature=temperature)
        return ChatGoogleGenerativeAI(
            model=target_model,
            api_key=api_key,
            google_api_key=api_key,
            temperature=temperature,
            n=1,
            max_retries=3,
        )

    logger.info("instantiating_cached_groq_client", model=target_model, temperature=temperature)
    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=target_model,
        temperature=temperature,
    )


def get_chat_model(
    provider: str = None,
    model_name: str = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Factory function returning cached Chat Models based on selected provider."""
    selected_provider = (provider or settings.llm_provider or "groq").lower().strip()

    if selected_provider in ("gemini", "google"):
        target_model = settings.gemini_model or "gemini-1.5-flash"
        if model_name and "gemini" in model_name.lower():
            target_model = model_name
        return _cached_chat_model("gemini", target_model, temperature)

    target_model = settings.groq_model or "llama-3.3-70b-versatile"
    if model_name and "llama" in model_name.lower():
        target_model = model_name
    return _cached_chat_model("groq", target_model, temperature)
