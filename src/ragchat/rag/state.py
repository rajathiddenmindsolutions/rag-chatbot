"""State definition for the RAG LangGraph orchestration."""

from typing import TypedDict
from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    """Orchestration state of the RAG pipeline."""

    query: str
    chunking_strategy: str
    expanded_queries: list[str]
    retrieved_chunks: list[Document]
    graded_chunks: list[Document]
    retry_count: int
    answer: str
    citations: list[dict]
    history: list[dict]
    query_type: str
