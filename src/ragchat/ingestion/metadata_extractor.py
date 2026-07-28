"""Metadata extractor module combining heuristics and LLM fallback (via Groq)."""

import json
from datetime import date
from typing import Optional
import structlog
from docling_core.types.doc import DoclingDocument
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from ragchat.config import settings

logger = structlog.get_logger(__name__)


class ExtractedMetadata(BaseModel):
    """Pydantic model representing structured document metadata."""

    title: str = Field(..., description="The main title of the document.")
    authors: list[str] = Field(default=[], description="List of authors of the document.")
    abstract: Optional[str] = Field(None, description="A brief abstract or summary of the document.")
    publication_date: Optional[str] = Field(
        None, description="The publication date or year if found, ideally as YYYY-MM-DD or YYYY."
    )


def extract_metadata_heuristically(doc: DoclingDocument) -> dict:
    """Extract metadata using simple heuristics (headings and description fields)."""
    title = None
    authors = []
    abstract = None

    # Check document description
    if hasattr(doc, "description") and doc.description:
        if getattr(doc.description, "title", None):
            title = doc.description.title

    # Try finding the first level 1 heading
    if not title and hasattr(doc, "iterate_items"):
        for item, level in doc.iterate_items():
            label = getattr(item, "label", "").lower()
            if label == "heading" and level == 1:
                title = getattr(item, "text", "")
                break

    # If title is still missing, use first element text as title fallback
    if not title and hasattr(doc, "iterate_items"):
        for item, level in doc.iterate_items():
            text = getattr(item, "text", "")
            if text:
                title = text[:100]
                break

    return {
        "title": title or "Untitled Document",
        "authors": authors,
        "abstract": abstract,
        "publication_date": None,
    }


async def extract_metadata_with_llm(text: str) -> Optional[ExtractedMetadata]:
    """Call Groq to extract structured metadata from the first page text."""
    if not settings.groq_api_key or settings.groq_api_key == "changeme":
        logger.warn("groq_api_key_not_configured_skipping_llm_metadata")
        return None

    try:
        # Initialize ChatGroq (we use llama-3.1-8b-instant as it is fast and cheap for simple extraction tasks)
        llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.0,
        )

        structured_llm = llm.with_structured_output(ExtractedMetadata)

        prompt = (
            "Analyze the following text from the first page of a document. "
            "Extract the document title, authors, brief abstract, and publication date "
            "if present. Return the response in the requested schema.\n\n"
            f"--- START TEXT ---\n{text[:3000]}\n--- END TEXT ---"
        )

        result = await structured_llm.ainvoke(prompt)
        return result
    except Exception as exc:
        logger.error("llm_metadata_extraction_failed", error=str(exc))
        return None


async def extract_metadata(doc: DoclingDocument, markdown_text: str) -> dict:
    """Extract metadata, starting with heuristics and falling back to Groq LLM if needed."""
    meta = extract_metadata_heuristically(doc)
    
    # If title looks generic or abstract/authors are missing, try LLM fallback
    if meta["title"] == "Untitled Document" or not meta["authors"] or not meta["abstract"]:
        logger.info("triggering_llm_metadata_extraction")
        llm_meta = await extract_metadata_with_llm(markdown_text)
        if llm_meta:
            meta["title"] = llm_meta.title or meta["title"]
            meta["authors"] = llm_meta.authors or meta["authors"]
            meta["abstract"] = llm_meta.abstract or meta["abstract"]
            if llm_meta.publication_date:
                # Try parsing publication date
                try:
                    pub_str = llm_meta.publication_date.strip()
                    if len(pub_str) == 4 and pub_str.isdigit():
                        meta["publication_date"] = date(int(pub_str), 1, 1)
                    else:
                        # try basic date parser
                        meta["publication_date"] = date.fromisoformat(pub_str[:10])
                except Exception:
                    meta["publication_date"] = None
                    
            logger.info("llm_metadata_extraction_success", title=meta["title"])
            
    return meta
