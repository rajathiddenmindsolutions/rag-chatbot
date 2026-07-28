"""Multi-layer RAG Caching Service.

Implements 3 caching layers:
1. ResponseCache: Full normalized (query + strategy + history) -> (answer, citations)
2. EmbeddingCache: Normalized query string -> list[float] vector
3. LLMPromptCache: SHA256(query + context_hash) -> answer string
"""

import hashlib
import json
import re
from collections import OrderedDict
from typing import Any, Optional
import structlog

logger = structlog.get_logger(__name__)


def normalize_text(text: str) -> str:
    """Normalize input text for cache matching (lowercase, strip, single spaces)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)     # collapse whitespace
    return text


class LRUCache:
    """Thread-safe-like LRU cache implementation with maxsize cap."""

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)  # Mark as recently used
        return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)  # Evict oldest entry

    def clear(self) -> None:
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)


class MultiLayerRAGCache:
    """Central cache manager providing Layer 1 (Response), Layer 2 (Embedding), Layer 3 (LLM Prompt)."""

    def __init__(self, max_response: int = 500, max_embed: int = 1000, max_llm: int = 500):
        self.response_cache = LRUCache(maxsize=max_response)
        self.embedding_cache = LRUCache(maxsize=max_embed)
        self.llm_prompt_cache = LRUCache(maxsize=max_llm)

    # ── Layer 1: Full Response Cache ─────────────────────────────────────────
    def _make_response_key(self, query: str, strategy: str, history: Optional[list] = None, provider: str = "groq") -> str:
        norm_q = normalize_text(query)
        hist_str = json.dumps(history or [], sort_keys=True)
        raw_key = f"resp:{norm_q}:{strategy}:{provider}:{hist_str}"
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def get_response(self, query: str, strategy: str, history: Optional[list] = None, provider: str = "groq") -> Optional[dict]:
        key = self._make_response_key(query, strategy, history, provider)
        hit = self.response_cache.get(key)
        if hit:
            logger.info("cache_hit_layer1_response", query=query, provider=provider)
        return hit

    def set_response(self, query: str, strategy: str, history: Optional[list], answer: str, citations: list, provider: str = "groq") -> None:
        key = self._make_response_key(query, strategy, history, provider)
        self.response_cache.set(key, {"answer": answer, "citations": citations})
        logger.info("cache_stored_layer1_response", query=query, provider=provider)

    # ── Layer 2: Embedding Cache ─────────────────────────────────────────────
    def get_embedding(self, query: str) -> Optional[list[float]]:
        key = f"emb:{normalize_text(query)}"
        hit = self.embedding_cache.get(key)
        if hit:
            logger.info("cache_hit_layer2_embedding", query=query)
        return hit

    def set_embedding(self, query: str, vector: list[float]) -> None:
        key = f"emb:{normalize_text(query)}"
        self.embedding_cache.set(key, vector)

    # ── Layer 3: LLM Prompt Cache ────────────────────────────────────────────
    def _make_prompt_key(self, query: str, context_str: str) -> str:
        norm_q = normalize_text(query)
        context_hash = hashlib.md5(context_str.encode()).hexdigest()
        return f"llm:{norm_q}:{context_hash}"

    def get_llm_response(self, query: str, context_str: str) -> Optional[str]:
        key = self._make_prompt_key(query, context_str)
        hit = self.llm_prompt_cache.get(key)
        if hit:
            logger.info("cache_hit_layer3_llm_prompt", query=query)
        return hit

    def set_llm_response(self, query: str, context_str: str, response: str) -> None:
        key = self._make_prompt_key(query, context_str)
        self.llm_prompt_cache.set(key, response)


# Global process singleton
rag_cache = MultiLayerRAGCache()
