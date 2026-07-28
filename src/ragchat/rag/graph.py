"""LangGraph orchestration graph wiring for ultra-low latency RAG chatbot.

Optimizations Applied:
1. Entry goes to classify_and_rephrase_node (LLM decides casual vs web vs search).
2. For "search": routes directly to retrieve_node (which does single-query retrieval first, with conditional expansion if <3 chunks).
3. Bypasses grade_documents_node for ultra-fast generation (relies on OpenSearch hybrid min_score cutoff).
"""

import structlog
from langgraph.graph import END, START, StateGraph

from ragchat.rag.nodes import (
    casual_response_node,
    classify_and_rephrase_node,
    generate_node,
    google_search_node,
    respond_not_found_node,
    retrieve_node,
)
from ragchat.rag.state import RAGState

logger = structlog.get_logger(__name__)


def route_start(state: RAGState) -> str:
    """Route based on LLM classification decision (casual vs search vs web)."""
    q_type = state.get("query_type", "search")
    if q_type == "casual":
        return "casual"
    elif q_type == "web":
        return "web"
    return "search"


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 1: Full RAG Graph (non-streaming /query endpoint)
# ═══════════════════════════════════════════════════════════════════════════════

builder = StateGraph(RAGState)

builder.add_node("classify_and_rephrase", classify_and_rephrase_node)
builder.add_node("casual_response", casual_response_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("google_search", google_search_node)
builder.add_node("generate", generate_node)
builder.add_node("respond_not_found", respond_not_found_node)

# Entry edge: ALWAYS start at classify_and_rephrase
builder.add_edge(START, "classify_and_rephrase")

# Route based on LLM decision: casual -> casual_response, web -> google_search, search -> retrieve
builder.add_conditional_edges(
    "classify_and_rephrase",
    route_start,
    {
        "casual": "casual_response",
        "web": "google_search",
        "search": "retrieve",
    },
)

builder.add_edge("retrieve", "generate")
builder.add_edge("google_search", "generate")
builder.add_edge("casual_response", END)
builder.add_edge("generate", END)
builder.add_edge("respond_not_found", END)

rag_graph = builder.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 2: Retrieval-only graph for streaming endpoint
# ═══════════════════════════════════════════════════════════════════════════════

retrieval_builder = StateGraph(RAGState)
retrieval_builder.add_node("classify_and_rephrase", classify_and_rephrase_node)
retrieval_builder.add_node("retrieve", retrieve_node)
retrieval_builder.add_node("google_search", google_search_node)

retrieval_builder.add_edge(START, "classify_and_rephrase")
retrieval_builder.add_conditional_edges(
    "classify_and_rephrase",
    route_start,
    {
        "casual": END,
        "web": "google_search",
        "search": "retrieve",
    },
)
retrieval_builder.add_edge("retrieve", END)
retrieval_builder.add_edge("google_search", END)

retrieval_graph = retrieval_builder.compile()
