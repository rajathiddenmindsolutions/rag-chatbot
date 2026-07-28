"""Standalone OpenAI-Compatible API Router for Open-WebUI integration.

Isolates all /v1/chat/completions and /v1/models endpoints into a single, clean file.
If ever needed, deleting this file cleanly removes all OpenAI-compatibility logic.
"""

import json
import time
from typing import Any, List, Optional
import structlog
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ragchat.api.routers.query import QueryRequest, stream_rag

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["OpenAI Compatibility"])


# ── OpenAI Schema Specifications ──────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="llama-3.3-70b-versatile")
    messages: List[ChatMessage]
    stream: bool = Field(default=True)
    temperature: Optional[float] = 0.7


# ── Models Endpoint (/v1/models) ──────────────────────────────────────────────

@router.get("/models")
async def list_models():
    """List custom models available for Open-WebUI dropdown selection."""
    return {
        "object": "list",
        "data": [
            {
                "id": "llama-3.3-70b-versatile",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "groq",
            },
            {
                "id": "gemini-3-flash-preview",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google",
            },
        ],
    }


# ── Chat Completions Endpoint (/v1/chat/completions) ─────────────────────────

@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """Bridge standard OpenAI /v1/chat/completions to internal RAG graph."""
    if not req.messages:
        return {"choices": [{"message": {"role": "assistant", "content": "No prompt received."}}]}

    # Extract last user prompt
    user_prompt = ""
    history_list = []
    for msg in req.messages:
        text_content = msg.content if isinstance(msg.content, str) else str(msg.content or "")
        if msg.role == "user":
            user_prompt = text_content
        history_list.append({"role": msg.role, "content": text_content})

    # ── Fast Intercept for Open-WebUI Background Title / Follow-Up Generation ──
    if "Generate a concise title" in user_prompt or "Generate 1-3 broad tags" in user_prompt:
        logger.info("intercepting_open_webui_title_generation", query="Title Generation")
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"title": "RAG Conversation"}'}, "finish_reason": "stop"}]
        }

    if "Suggest 3-5 relevant follow-up questions" in user_prompt:
        logger.info("intercepting_open_webui_followup_generation", query="Follow-up Generation")
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"follow_ups": []}'}, "finish_reason": "stop"}]
        }

    # Resolve provider
    selected_provider = "gemini" if "gemini" in req.model.lower() else "groq"

    # Construct internal QueryRequest
    internal_req = QueryRequest(
        query=user_prompt,
        chunking_strategy="structural",
        provider=selected_provider,
        history=history_list[:-1] if len(history_list) > 1 else [],
    )

    # ── Non-Streaming Response ────────────────────────────────────────────────
    if not req.stream:
        logger.info("openai_compat_non_streaming_query", query=user_prompt, provider=selected_provider)
        # Call internal RAG pipeline logic
        accumulated_text = ""
        sse_generator = (await stream_rag(internal_req)).body_iterator
        async for chunk in sse_generator:
            chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            for line in chunk_str.split("\n"):
                if line.startswith("data: ") and not line.startswith("data: ["):
                    accumulated_text += line[6:]

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": accumulated_text},
                "finish_reason": "stop"
            }]
        }

    # ── Streaming SSE Response (OpenAI Delta Format) ──────────────────────────
    logger.info("openai_compat_streaming_query", query=user_prompt, provider=selected_provider)

    async def openai_sse_generator():
        created_time = int(time.time())
        model_name = req.model

        # Fetch internal stream iterator
        stream_resp = await stream_rag(internal_req)
        async for raw_chunk in stream_resp.body_iterator:
            chunk_str = raw_chunk.decode("utf-8") if isinstance(raw_chunk, bytes) else str(raw_chunk)
            lines = chunk_str.split("\n\n")
            for line in lines:
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                content = line[6:]
                if content == "[DONE]":
                    stop_payload = {
                        "id": f"chatcmpl-{created_time}",
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(stop_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                elif not content.startswith("[CITATIONS]"):
                    delta_payload = {
                        "id": f"chatcmpl-{created_time}",
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(delta_payload)}\n\n"

    return StreamingResponse(
        openai_sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
