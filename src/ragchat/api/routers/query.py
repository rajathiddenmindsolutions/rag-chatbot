"""API Router for executing RAG queries — 100% LLM-Driven Classification & SSE Streaming."""

import asyncio
import json
import re
import time
from typing import Any
import structlog
from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from ragchat.config import settings
from ragchat.eval.evaluator import evaluate_rag_async
from ragchat.cache.rag_cache import rag_cache
from ragchat.llm.chat_models import get_chat_model
from ragchat.rag.graph import rag_graph, retrieval_graph
from ragchat.rag.nodes import (
    build_generation_inputs,
    _build_langchain_history,
)
from ragchat.rag.prompt_templates import CASUAL_PROMPT, GENERATION_PROMPT
from ragchat.storage.schemas import Citation, QueryRequest, QueryResponse

logger = structlog.get_logger(__name__)
router = APIRouter()


def _extract_text_content(content: Any) -> str:
    """Extract plain string text from chunk.content (handles str, list of dicts, or list of str from Gemini)."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts)
    return str(content or "")


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Execute a full RAG query through the 100% LLM-driven graph."""
    logger.info(
        "received_rag_query",
        query=request.query,
        strategy=request.chunking_strategy,
        history_len=len(request.history),
    )
    start_time = time.time()

    # ── Layer 1 Cache Check (~1ms) ───────────────────────────────────────────
    cached_resp = rag_cache.get_response(request.query, request.chunking_strategy, request.history)
    if cached_resp:
        latency = time.time() - start_time
        citations = [Citation(**c) for c in cached_resp["citations"]]
        return QueryResponse(
            query=request.query,
            answer=cached_resp["answer"],
            citations=citations,
            latency_seconds=latency,
        )

    try:
        result_state = await rag_graph.ainvoke({
            "query": request.query,
            "chunking_strategy": request.chunking_strategy,
            "history": request.history,
            "retry_count": 0,
        })

        latency = time.time() - start_time
        answer = result_state.get("answer", "No response generated.")
        raw_citations = result_state.get("citations", [])

        # Store in Layer 1 Cache
        rag_cache.set_response(
            query=request.query,
            strategy=request.chunking_strategy,
            history=request.history,
            answer=answer,
            citations=raw_citations,
        )

        citations = [
            Citation(
                document_id=c["document_id"],
                title=c.get("title") or "Unknown Document",
                section_path=c.get("section_path") or "root",
                chunk_index=c.get("chunk_index", 0),
                text=c["text"],
            )
            for c in raw_citations
        ]

        logger.info("rag_query_success", latency_seconds=latency)
        return QueryResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            latency_seconds=latency,
        )
    except Exception as exc:
        logger.error("rag_query_failed", error=str(exc))
        return QueryResponse(
            query=request.query,
            answer=f"An error occurred while answering your question: {exc}",
            citations=[],
            latency_seconds=time.time() - start_time,
        )


@router.post("/stream")
async def stream_rag(request: QueryRequest):
    """Streaming SSE endpoint — 100% LLM classification.

    Flow:
      1. Run retrieval_graph (which executes parallel classify_and_rephrase_node).
      2. If LLM classified as "casual" -> stream casual LLM response directly.
      3. If LLM classified as "search"  -> stream RAG generation from retrieved chunks.
    """
    logger.info(
        "received_stream_query",
        query=request.query,
        strategy=request.chunking_strategy,
        history_len=len(request.history),
    )

    async def event_generator():
        start_time = time.time()
        provider = request.provider or "groq"

        # ── Layer 1 Cache Check for Stream (~1ms) ────────────────────────────
        cached_resp = rag_cache.get_response(request.query, request.chunking_strategy, request.history, provider=provider)
        if cached_resp:
            yield f"data: {cached_resp['answer']}\n\n"
            citations_payload = json.dumps(cached_resp["citations"])
            yield f"data: [CITATIONS] {citations_payload}\n\n"
            yield "data: [DONE]\n\n"
            return

        accumulated_answer = ""

        try:
            # Run graph up to retrieval (LLM handles classify + rephrase in parallel)
            retrieval_state = await retrieval_graph.ainvoke({
                "query": request.query,
                "chunking_strategy": request.chunking_strategy,
                "provider": provider,
                "history": request.history,
                "retry_count": 0,
            })

            query_type = retrieval_state.get("query_type", "search")
            rephrased_query = retrieval_state.get("query", request.query)

            # ── Case A: LLM decided CASUAL (general question / smalltalk) ────
            if query_type == "casual":
                logger.info("stream_handling_casual", query=request.query, provider=provider)
                langchain_history = _build_langchain_history(request.history or [])
                llm = get_chat_model(provider=provider, temperature=0.7)
                chain = CASUAL_PROMPT | llm
                async for chunk in chain.astream({
                    "chat_history": langchain_history,
                    "question": request.query,
                }):
                    text_chunk = _extract_text_content(chunk.content)
                    if text_chunk:
                        accumulated_answer += text_chunk
                        yield f"data: {text_chunk}\n\n"

                yield "data: [CITATIONS] []\n\n"
                rag_cache.set_response(request.query, request.chunking_strategy, request.history, accumulated_answer, [], provider=provider)
                latency = time.time() - start_time
                logger.info("stream_casual_success", provider=provider, latency_seconds=round(latency, 3))
                yield "data: [DONE]\n\n"
                return

            # ── Case B: Check retrieved or graded chunks ──
            chunks = retrieval_state.get("retrieved_chunks") or retrieval_state.get("graded_chunks", [])
            if not chunks:
                logger.info("stream_search_0_chunks_fallback_to_general_knowledge", query=rephrased_query, provider=provider)
                langchain_history = _build_langchain_history(request.history or [])
                llm = get_chat_model(provider=provider, temperature=0.7)
                chain = CASUAL_PROMPT | llm
                async for chunk in chain.astream({
                    "chat_history": langchain_history,
                    "question": rephrased_query,
                }):
                    text_chunk = _extract_text_content(chunk.content)
                    if text_chunk:
                        accumulated_answer += text_chunk
                        yield f"data: {text_chunk}\n\n"

                yield "data: [CITATIONS] []\n\n"
                rag_cache.set_response(request.query, request.chunking_strategy, request.history, accumulated_answer, [], provider=provider)
                latency = time.time() - start_time
                logger.info("stream_fallback_success", provider=provider, latency_seconds=round(latency, 3))
                yield "data: [DONE]\n\n"
                return

            context_str, langchain_history, citations = build_generation_inputs(retrieval_state)

            # Layer 3 Cache Check (LLM Prompt Cache)
            cached_llm = rag_cache.get_llm_response(rephrased_query, context_str)
            if cached_llm:
                yield f"data: {cached_llm}\n\n"
                citations_payload = json.dumps(citations)
                yield f"data: [CITATIONS] {citations_payload}\n\n"
                rag_cache.set_response(request.query, request.chunking_strategy, request.history, cached_llm, citations, provider=provider)
                latency = time.time() - start_time
                logger.info("stream_cache_layer3_success", provider=provider, latency_seconds=round(latency, 3))
                yield "data: [DONE]\n\n"
                return

            llm = get_chat_model(provider=provider, temperature=0.0)
            chain = GENERATION_PROMPT | llm

            async for chunk in chain.astream({
                "query": rephrased_query,
                "context": context_str,
                "chat_history": langchain_history,
            }):
                token = _extract_text_content(chunk.content)
                if token:
                    clean = re.sub(r'\s*\[Doc:[^\]]+\]', '', token)
                    if clean:
                        accumulated_answer += clean
                        yield f"data: {clean}\n\n"

            citations_payload = json.dumps(citations)
            yield f"data: [CITATIONS] {citations_payload}\n\n"

            # Save in Layer 1 and Layer 3
            rag_cache.set_llm_response(rephrased_query, context_str, accumulated_answer)
            rag_cache.set_response(request.query, request.chunking_strategy, request.history, accumulated_answer, citations, provider=provider)

            latency = time.time() - start_time
            logger.info("stream_rag_success", provider=provider, latency_seconds=round(latency, 3))

            # Dispatch Unblocking Background Ragas Evaluation (0ms latency impact)
            if accumulated_answer and retrieval_state.get("retrieved_chunks"):
                context_texts = [getattr(c, "page_content", getattr(c, "text", str(c))) for c in retrieval_state["retrieved_chunks"]]
                asyncio.create_task(evaluate_rag_async(
                    query=request.query,
                    context_chunks=context_texts,
                    answer=accumulated_answer,
                    provider=provider,
                ))

            yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.error("stream_query_failed", error=str(exc))
            yield f"data: [ERROR] {exc}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
