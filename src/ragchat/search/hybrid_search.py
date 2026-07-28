"""Hybrid search (BM25 + k-NN) with Reciprocal Rank Fusion (RRF) for OpenSearch."""

import asyncio
import structlog
from opensearchpy import AsyncOpenSearch

from ragchat.config import settings

logger = structlog.get_logger(__name__)


async def search_bm25(
    client: AsyncOpenSearch,
    query_text: str,
    strategy: str = None,
    size: int = 30,
) -> list[dict]:
    """Execute a text-based BM25 query on the chunks index.

    NOTE: ``strategy`` is intentionally NOT applied as a filter here anymore.
    Chunking strategy is an ingestion-time choice (how a document was split),
    not a query-time relevance signal. Filtering search results by the
    currently-selected UI strategy meant documents ingested under a
    different strategy became permanently invisible to search, regardless
    of relevance — a document could be fully indexed and topically perfect
    for a query and never surface. Search now always runs across the whole
    corpus. The parameter is kept (unused) for backward compatibility with
    callers; remove once all call sites are updated.
    """
    body = {
        "query": {"match": {"text": query_text}},
        "size": size,
    }

    try:
        response = await client.search(index=settings.opensearch_index, body=body)
        hits = response["hits"]["hits"]
        return hits
    except Exception as exc:
        logger.error("bm25_search_failed", error=str(exc))
        return []


async def search_knn(
    client: AsyncOpenSearch,
    query_vector: list[float],
    strategy: str = None,
    size: int = 30,
) -> list[dict]:
    """Execute a k-NN vector search query on the chunks index.

    NOTE: ``strategy`` is intentionally NOT applied as a filter — see
    search_bm25's docstring for why. Search now always runs across the
    whole corpus regardless of chunking strategy.
    """
    knn_clause = {
        "embedding": {
            "vector": query_vector,
            "k": size,
        }
    }
    body = {
        "query": {"knn": knn_clause},
        "size": size,
    }

    try:
        response = await client.search(index=settings.opensearch_index, body=body)
        hits = response["hits"]["hits"]
        return hits
    except Exception as exc:
        logger.error("knn_search_failed", error=str(exc))
        return []


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    knn_results: list[dict],
    k_rrf: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """Combine text and vector search results using Reciprocal Rank Fusion (RRF)."""
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    def add_results(results: list[dict]):
        for rank, hit in enumerate(results, 1):
            doc_id = hit["_id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = hit["_source"]
            
            # Update RRF score
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + rank))

    add_results(bm25_results)
    add_results(knn_results)

    # Sort doc_ids by their descending RRF score
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    fused_results = []
    for doc_id in sorted_doc_ids[:top_k]:
        source = doc_map[doc_id]
        fused_results.append({
            "chunk_id": doc_id,
            "document_id": source["document_id"],
            "chunk_index": source["chunk_index"],
            "text": source["text"],
            "section_path": source.get("section_path") or "",
            "title": source.get("title") or "",
            "authors": source.get("authors") or [],
            "chunking_strategy": source["chunking_strategy"],
            "score": rrf_scores[doc_id],
        })

    return fused_results


async def hybrid_search(
    client: AsyncOpenSearch,
    query_text: str,
    query_vector: list[float],
    strategy: str = None,
    top_k: int = 10,
    min_score: float = 0.015,
) -> list[dict]:
    """Perform hybrid search (BM25 + k-NN) combined with Reciprocal Rank Fusion (RRF).

    ``min_score`` is a cheap first-pass noise filter on the fused RRF score.
    With the default k_rrf=60, a doc ranked #1 in a single result list scores
    ~0.0164, and a doc ranked highly in BOTH lists scores ~0.03+. Anything
    below ~0.015 is typically a weak, coincidental keyword/vector match rather
    than a genuinely relevant chunk, so we drop it here before it ever reaches
    generation. This is intentionally conservative — it does not replace the
    LLM-based relevance grading in grade_documents_node, it just avoids paying
    for/generating from chunks that are obviously irrelevant.
    """
    # Fetch more than top_k for RRF to work effectively
    fetch_size = max(top_k * 3, 30)

    # Run BM25 and kNN queries in parallel
    bm25_hits, knn_hits = await asyncio.gather(
        search_bm25(client, query_text, strategy=strategy, size=fetch_size),
        search_knn(client, query_vector, strategy=strategy, size=fetch_size),
    )

    fused = reciprocal_rank_fusion(bm25_hits, knn_hits, top_k=top_k)

    filtered = [hit for hit in fused if hit["score"] >= min_score]
    dropped = len(fused) - len(filtered)
    if dropped:
        logger.info(
            "hybrid_search_filtered_low_relevance",
            query=query_text,
            dropped_count=dropped,
            min_score=min_score,
        )

    logger.info("hybrid_search_completed", query=query_text, results_count=len(filtered))
    return filtered
