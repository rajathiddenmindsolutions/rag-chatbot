"""Node implementations for the RAG LangGraph orchestration."""

import asyncio
import json
from typing import Optional
import structlog
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from ragchat.llm.chat_models import get_chat_model
from ragchat.rag.prompt_templates import (
    CASUAL_PROMPT,
    DECISION_PROMPT_TEMPLATE,
    DOCUMENT_GRADER_PROMPT,
    GENERATION_PROMPT,
    QUERY_EXPANSION_PROMPT,
    REPHRASE_PROMPT,
)
from ragchat.rag.retriever import OpenSearchHybridRetriever
from ragchat.rag.state import RAGState
from ragchat.search.embeddings import LocalEmbeddings
from ragchat.search.opensearch_client import get_opensearch_client

logger = structlog.get_logger(__name__)


class GradeScore(BaseModel):
    """Schema for document grading result."""

    score: str = Field(description="Relevance score, must be either 'yes' or 'no'.")


_REPHRASE_LEAKAGE_MARKERS = (
    "standalone search query",
    "standalone question",
    "rephrase the",
    "rephrased question",
    "rewrite the latest",
    "conversation history",
    "please provide the question",
    "i don't see a specific question",
    "as a language model",
)


def _looks_like_meta_leakage(text: str) -> bool:
    """Detect rephrase-LLM output that echoes prompt/meta instructions rather
    than a clean standalone query. Small models occasionally do this when the
    input is itself vague or conversational (e.g. 'what kind of question am I
    asking?'), producing text like 'please provide the question you'd like me
    to rephrase' instead of an actual rephrased query. Long, sentence-like
    outputs are also suspicious since REPHRASE_SYSTEM caps results at ~20 words.
    """
    lowered = text.lower()
    if any(marker in lowered for marker in _REPHRASE_LEAKAGE_MARKERS):
        return True
    if len(text.split()) > 30:
        return True
    return False


def _build_langchain_history(history: list[dict]) -> list:
    """Convert list of {role, content} dicts into LangChain message objects."""
    messages = []
    for msg in history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


# ---------------------------------------------------------------------------
# classify_and_rephrase_node
# Runs classification AND query rephrasing IN PARALLEL with asyncio.gather().
# This saves ~400ms vs running them serially.
# ---------------------------------------------------------------------------

async def classify_and_rephrase_node(state: RAGState) -> RAGState:
    """Classify query type AND rephrase for context-awareness — both in parallel."""
    query = state["query"]
    history = state.get("history") or []
    langchain_history = _build_langchain_history(history)

    provider = state.get("provider")

    # ── Fast local short-circuit for pure greetings (no LLM call at all) ──
    greetings = {
        "hi", "hii", "hello", "hey", "hola", "greetings",
        "good morning", "good afternoon", "good evening", "sup", "yo",
        "howdy", "hi there", "hey there",
    }
    cleaned_query = query.strip().lower().rstrip("?.! ")
    if cleaned_query in greetings:
        logger.info("query_classified_locally", query_type="casual")
        return {**state, "query_type": "casual", "query": query}

    # ── Define the two async tasks ──────────────────────────────────────────

    async def classify_task() -> str:
        """Call LLM to route query: RETRIEVE or RESPOND."""
        llm = get_chat_model(provider=provider, temperature=0.0)
        prompt = DECISION_PROMPT_TEMPLATE.format(question=query)
        try:
            res = await llm.ainvoke(prompt)
            result = res.content.strip().upper()
            if "WEB" in result:
                return "web"
            elif "RESPOND" in result:
                return "casual"
            else:
                return "search"
        except Exception as exc:
            logger.warning("query_classification_failed", error=str(exc))
            return "search"  # safe default

    async def rephrase_task() -> str:
        """Rephrase the query into a standalone question if history exists."""
        if not langchain_history:
            return query  # Already standalone — skip LLM call entirely
        try:
            rephrase_llm = get_chat_model(provider=provider, temperature=0.0)
            rephrase_chain = REPHRASE_PROMPT | rephrase_llm
            res = await rephrase_chain.ainvoke({
                "chat_history": langchain_history,
                "question": query,
            })
            rephrased = res.content.strip()

            if _looks_like_meta_leakage(rephrased):
                logger.warning(
                    "rephrase_output_looks_like_meta_leakage_falling_back",
                    original=query,
                    rejected_output=rephrased,
                )
                return query

            logger.info("rephrased_query", standalone_query=rephrased)
            return rephrased
        except Exception as exc:
            logger.warning("query_rephrasing_failed", error=str(exc))
            return query  # Fallback to original

    # ── Fire both tasks in parallel — the key latency saving ───────────────
    query_type, rephrased_query = await asyncio.gather(
        classify_task(),
        rephrase_task(),
    )
    logger.info(
        "classify_and_rephrase_done",
        query_type=query_type,
        original=query,
        rephrased=rephrased_query,
    )
    return {**state, "query_type": query_type, "query": rephrased_query}


# ---------------------------------------------------------------------------
# expand_query_node  (rephrasing now done upstream — only expansion here)
# ---------------------------------------------------------------------------

async def expand_query_node(state: RAGState) -> RAGState:
    """Generate 2 alternative query phrasings to increase retrieval recall."""
    query = state["query"]  # Already rephrased by classify_and_rephrase_node
    logger.info("expanding_query", query=query)

    provider = state.get("provider")
    llm = get_chat_model(provider=provider, temperature=0.2)
    chain = QUERY_EXPANSION_PROMPT | llm

    try:
        response = await chain.ainvoke({"query": query})
        text = response.content.strip()
        variations = [line.strip() for line in text.split("\n") if line.strip()]
        cleaned_variations = [
            var.lstrip("0123456789.-*• ")
            for var in variations
            if var.lstrip("0123456789.-*• ")
        ]
        all_queries = [query] + cleaned_variations[:2]
        logger.info("query_expansion_completed", queries=all_queries)
        return {**state, "query": query, "expanded_queries": all_queries}
    except Exception as exc:
        logger.error("query_expansion_failed", error=str(exc))
        return {**state, "query": query, "expanded_queries": [query]}


# ---------------------------------------------------------------------------
# retrieve_node
# ---------------------------------------------------------------------------

async def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve document chunks. Performs fast single-query retrieval first, only expanding if <3 chunks returned."""
    primary_query = state["query"]
    strategy = state.get("chunking_strategy", "structural")
    provider = state.get("provider")

    client = get_opensearch_client()
    embeddings = LocalEmbeddings()
    retriever = OpenSearchHybridRetriever(
        client=client,
        embeddings_model=embeddings,
        chunking_strategy=strategy,
        top_k=5,
    )

    try:
        # Step 1: Fast initial retrieval with primary query
        initial_chunks = await retriever.ainvoke(primary_query)

        # Step 2: Conditional Query Expansion (Paid ONLY if initial retrieval is thin)
        if len(initial_chunks) < 3:
            logger.info("thin_initial_retrieval_triggering_query_expansion", count=len(initial_chunks), query=primary_query)
            try:
                llm = get_chat_model(provider=provider, temperature=0.2)
                chain = QUERY_EXPANSION_PROMPT | llm
                response = await chain.ainvoke({"query": primary_query})
                text = response.content.strip()
                variations = [
                    line.strip().lstrip("0123456789.-*• ")
                    for line in text.split("\n")
                    if line.strip()
                ]
                extra_queries = [v for v in variations if v][:2]
                if extra_queries:
                    extra_results = await asyncio.gather(*[retriever.ainvoke(q) for q in extra_queries])
                    for doc_list in extra_results:
                        initial_chunks.extend(doc_list)
            except Exception as exp_exc:
                logger.warning("conditional_expansion_skipped", error=str(exp_exc))

        # Deduplicate results
        seen_ids = set()
        deduped_chunks = []
        for doc in initial_chunks:
            chunk_id = doc.metadata["chunk_id"]
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                deduped_chunks.append(doc)

        final_chunks = deduped_chunks[:5]

        # ── Serper.dev Google Search Tool Fallback ──────────────────────────────
        if not final_chunks:
            from ragchat.search.web_search import search_google_serper
            from langchain_core.documents import Document

            logger.info("opensearch_0_chunks_triggering_google_serper_search", query=primary_query)
            web_results = await search_google_serper(primary_query)

            if web_results:
                web_chunks = [
                    Document(
                        page_content=item["text"],
                        metadata={
                            "document_id": item["link"],
                            "chunk_id": item["link"],
                            "title": f"🌐 {item['title']}",
                            "section_path": item["link"],
                            "chunk_index": 0,
                            "chunking_strategy": "web_search",
                            "score": 1.0,
                        }
                    )
                    for item in web_results
                ]
                final_chunks = web_chunks
                logger.info("web_search_chunks_assembled", count=len(final_chunks))

        logger.info("retrieval_completed", total_retrieved=len(deduped_chunks), final_kept=len(final_chunks))

        await client.close()
        return {**state, "retrieved_chunks": final_chunks}
    except Exception as exc:
        logger.error("retrieval_failed", error=str(exc))
        await client.close()
        return {**state, "retrieved_chunks": []}


# ---------------------------------------------------------------------------
# google_search_node  (First-Class Tool Node for Real-World Web Search)
# ---------------------------------------------------------------------------

async def google_search_node(state: RAGState) -> RAGState:
    """Dedicated LangGraph node to perform real-time Google Search using Serper.dev."""
    query = state.get("rephrased_query") or state["query"]
    logger.info("executing_google_search_node", query=query)

    from ragchat.search.web_search import search_google_serper
    from langchain_core.documents import Document

    web_results = await search_google_serper(query, num_results=5)

    if web_results:
        web_chunks = [
            Document(
                page_content=item["text"],
                metadata={
                    "document_id": item["link"],
                    "chunk_id": item["link"],
                    "title": f"🌐 {item['title']}",
                    "section_path": item["link"],
                    "chunk_index": 0,
                    "chunking_strategy": "web_search",
                    "score": 1.0,
                }
            )
            for item in web_results
        ]
        logger.info("google_search_node_success", count=len(web_chunks))
        return {**state, "retrieved_chunks": web_chunks}
    else:
        logger.warning("google_search_node_no_results", query=query)
        return {**state, "retrieved_chunks": []}


# ---------------------------------------------------------------------------
# grade_documents_node  (kept but bypassed in graph for performance)
# ---------------------------------------------------------------------------

async def grade_documents_node(state: RAGState) -> RAGState:
    """Grade retrieved chunks for relevance to filter out noise."""
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    logger.info("grading_retrieved_documents", chunk_count=len(chunks))

    if not chunks:
        return {**state, "graded_chunks": []}

    provider = state.get("provider")
    llm = get_chat_model(provider=provider, temperature=0.0)
    grader_chain = DOCUMENT_GRADER_PROMPT | llm

    async def grade_single_doc(doc: Document) -> Optional[Document]:
        try:
            res = await grader_chain.ainvoke({"query": query, "document": doc.page_content})
            text = res.content.strip().lower()
            if "yes" in text or "binary_score: yes" in text or '"binary_score": "yes"' in text:
                return doc
        except Exception as exc:
            logger.warning("doc_grading_failed_retaining_chunk", error=str(exc))
            return doc
        return None

    tasks = [grade_single_doc(doc) for doc in chunks]
    graded_results = await asyncio.gather(*tasks)
    relevant_chunks = [doc for doc in graded_results if doc is not None]
    logger.info("grading_completed", relevant_count=len(relevant_chunks), original_count=len(chunks))

    return {**state, "graded_chunks": relevant_chunks}


# ---------------------------------------------------------------------------
# build_generation_inputs
# Shared helper used by both generate_node and the streaming endpoint.
# ---------------------------------------------------------------------------

def build_generation_inputs(state: RAGState) -> tuple[str, list, list]:
    """Extract context string, langchain_history, and citations from state.

    Returns:
        (context_str, langchain_history, citations)
    """
    query = state["query"]
    chunks = state.get("graded_chunks") or state.get("retrieved_chunks", [])
    history = state.get("history") or []

    context_blocks = []
    citations = []

    for idx, doc in enumerate(chunks, 1):
        meta = doc.metadata
        title = meta.get("title") or "Unknown Document"
        section = meta.get("section_path") or "root"
        chunk_idx = meta.get("chunk_index", 0)

        context_blocks.append(
            f"--- CHUNK {idx} (Source: {title} > {section} [{chunk_idx}]) ---\n"
            f"{doc.page_content}"
        )
        citations.append({
            "document_id": meta["document_id"],
            "title": title,
            "section_path": section,
            "chunk_index": chunk_idx,
            "text": doc.page_content,
        })

    context_str = "\n\n".join(context_blocks)
    langchain_history = _build_langchain_history(history)
    return context_str, langchain_history, citations


# ---------------------------------------------------------------------------
# generate_node  (non-streaming — used in the standard /query endpoint)
# ---------------------------------------------------------------------------

async def generate_node(state: RAGState) -> RAGState:
    """Generate final response text from retrieved chunks (non-streaming)."""
    query = state["query"]
    chunks = state.get("graded_chunks") or state.get("retrieved_chunks", [])
    logger.info("generating_answer", chunk_count=len(chunks))

    if not chunks:
        return {**state, "answer": "I'm sorry, I could not find enough relevant context to answer your question."}

    context_str, langchain_history, citations = build_generation_inputs(state)

    llm = get_chat_model(temperature=0.0)
    chain = GENERATION_PROMPT | llm

    try:
        response = await chain.ainvoke({
            "query": query,
            "context": context_str,
            "chat_history": langchain_history,
        })
        answer = response.content.strip()
        logger.info("generation_completed")
        return {**state, "answer": answer, "citations": citations}
    except Exception as exc:
        logger.error("generation_failed", error=str(exc))
        return {**state, "answer": f"Error generating response: {exc}", "citations": []}


# ---------------------------------------------------------------------------
# respond_not_found_node
# ---------------------------------------------------------------------------

async def respond_not_found_node(state: RAGState) -> RAGState:
    """Fallback when 0 document chunks are found — answer using LLM general knowledge."""
    query = state["query"]
    history = state.get("history") or []
    logger.info("triggering_general_knowledge_fallback", query=query)

    langchain_history = _build_langchain_history(history)
    llm = get_chat_model(temperature=0.7)
    chain = CASUAL_PROMPT | llm

    try:
        response = await chain.ainvoke({
            "chat_history": langchain_history,
            "question": f"{query} (Note: Provide a helpful general answer as no matching context was found in the uploaded documents)",
        })
        answer = response.content.strip()
        return {**state, "answer": answer, "citations": []}
    except Exception as exc:
        logger.error("general_knowledge_fallback_failed", error=str(exc))
        return {**state, "answer": "I could not find matching information in your uploaded documents. Please try rephrasing your question or checking your chunking strategy.", "citations": []}


# ---------------------------------------------------------------------------
# casual_response_node
# ---------------------------------------------------------------------------

async def casual_response_node(state: RAGState) -> RAGState:
    """Respond naturally to greetings, small talk, or general knowledge questions."""
    query = state["query"]
    history = state.get("history") or []
    logger.info("handling_casual_conversation")

    langchain_history = _build_langchain_history(history)

    llm = get_chat_model(temperature=0.7)
    chain = CASUAL_PROMPT | llm

    try:
        response = await chain.ainvoke({
            "chat_history": langchain_history,
            "question": query,
        })
        answer = response.content.strip()
        return {**state, "answer": answer, "citations": []}
    except Exception as exc:
        logger.error("casual_response_failed", error=str(exc))
        return {**state, "answer": "Hello! How can I help you today?", "citations": []}


# ---------------------------------------------------------------------------
# fast_retrieve_and_generate_node
# FAST PATH: Skips classify, rephrase, and expand.
# Goes directly: embed → search → generate (1 LLM call only).
# Uses the DEFAULT model configured in settings — not a forced smaller model.
# ---------------------------------------------------------------------------

async def fast_retrieve_and_generate_node(state: RAGState) -> RAGState:
    """Fast path: embed → search → generate. Skips all LLM preprocessing."""
    query = state["query"]
    strategy = state.get("chunking_strategy", "structural")
    history = state.get("history") or []
    logger.info("fast_path_triggered", query=query, strategy=strategy)

    # ── Step 1: Retrieve (embed + hybrid search — no query expansion) ──────
    client = get_opensearch_client()
    embeddings = LocalEmbeddings()
    retriever = OpenSearchHybridRetriever(
        client=client,
        embeddings_model=embeddings,
        chunking_strategy=strategy,
        top_k=5,
    )

    try:
        chunks = await retriever.ainvoke(query)
        await client.close()
    except Exception as exc:
        logger.error("fast_path_retrieval_failed", error=str(exc))
        await client.close()
        return {
            **state,
            "query_type": "fast",
            "answer": "I encountered an error while searching. Please try again.",
            "citations": [],
        }

    if not chunks:
        logger.info("fast_path_no_chunks_found_fallback_to_casual")
        return await casual_response_node(state)

    logger.info("fast_path_retrieval_done", chunks_found=len(chunks))

    # ── Step 2: Generate (default model, no forced 8b) ─────────────────────
    fast_state = {**state, "retrieved_chunks": chunks, "query_type": "fast"}
    context_str, langchain_history, citations = build_generation_inputs(fast_state)

    llm = get_chat_model(temperature=0.0)  # uses default model from settings
    chain = GENERATION_PROMPT | llm

    try:
        response = await chain.ainvoke({
            "query": query,
            "context": context_str,
            "chat_history": langchain_history,
        })
        answer = response.content.strip()
        logger.info("fast_path_generation_completed")
        return {**fast_state, "answer": answer, "citations": citations}
    except Exception as exc:
        logger.error("fast_path_generation_failed", error=str(exc))
        return {**fast_state, "answer": f"Error generating response: {exc}", "citations": []}
