"""Asynchronous, zero-latency RAG Evaluation module (n=1 API Compliant).

Runs in background tasks with 0ms impact on user response times.
Evaluates:
- Faithfulness (Hallucination score: 0.0 to 1.0)
- Answer Relevancy (Question-Answer match score: 0.0 to 1.0)
"""

import asyncio
import json
import re
from typing import List, Optional
import structlog

from ragchat.llm.chat_models import get_chat_model

logger = structlog.get_logger(__name__)

EVALUATION_PROMPT = """You are an expert RAG System Evaluator. Assess the generated answer against the user query and retrieved context chunks.

### USER QUERY:
{query}

### RETRIEVED CONTEXT CHUNKS:
{context}

### GENERATED ANSWER:
{answer}

### INSTRUCTIONS:
Evaluate the following 2 metrics with a floating-point score between 0.0 and 1.0:
1. "faithfulness": (0.0 to 1.0) Score 1.0 if ALL statements in the answer are fully supported by the retrieved context. Score lower if the answer introduces unsupported claims.
2. "answer_relevance": (0.0 to 1.0) Score 1.0 if the answer directly and completely answers the user query. Score lower if off-topic.

### OUTPUT FORMAT:
Return STRICTLY a JSON object in this exact schema:
{{
  "faithfulness": 0.95,
  "answer_relevance": 1.0
}}"""


async def evaluate_rag_async(
    query: str,
    context_chunks: List[str],
    answer: str,
    provider: str = "groq",
    model_name: Optional[str] = None,
):
    """Executes evaluation in an unblocking background task with n=1 API compliance."""
    if not query or not answer or not context_chunks:
        logger.warning("eval_skipped", reason="missing_inputs")
        return

    try:
        logger.info("starting_background_eval", query=query, provider=provider)

        context_str = "\n\n".join(f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks[:5]))
        formatted_prompt = EVALUATION_PROMPT.format(
            query=query,
            context=context_str,
            answer=answer,
        )

        target_model = model_name or ("llama-3.3-70b-versatile" if provider == "groq" else "gemini-1.5-flash")
        llm = get_chat_model(provider=provider, model_name=target_model, temperature=0.0)

        # Execute 1 single LLM-as-a-Judge call (n=1 compliant)
        eval_resp = await llm.ainvoke(formatted_prompt)
        resp_text = eval_resp.content if isinstance(eval_resp.content, str) else str(eval_resp.content or "")

        json_match = re.search(r"\{.*\}", resp_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            f_score = float(data.get("faithfulness", 0.0))
            r_score = float(data.get("answer_relevance", 0.0))

            logger.info(
                "ragas_evaluation_completed",
                query=query,
                faithfulness_score=round(f_score, 4),
                answer_relevancy_score=round(r_score, 4),
                provider=provider,
            )
        else:
            logger.warning("eval_parse_failed", raw_output=resp_text[:200])

    except Exception as exc:
        logger.error("eval_failed", error=str(exc))
