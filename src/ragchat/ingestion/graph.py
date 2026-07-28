"""LangGraph orchestration graph for document ingestion."""

from pathlib import Path
from typing import Optional, TypedDict
from uuid import UUID, uuid4

import structlog
from docling_core.types.doc import DoclingDocument
from langgraph.graph import END, START, StateGraph

from ragchat.chunking.base import ChunkResult
from ragchat.chunking.fixed_size import FixedSizeChunker
from ragchat.chunking.recursive import RecursiveMarkdownChunker
from ragchat.chunking.semantic import SemanticChunker
from ragchat.chunking.structural import StructuralChunker
from ragchat.ingestion.docling_parser import parse_pdf_to_docling
from ragchat.ingestion.metadata_extractor import extract_metadata
from ragchat.search.embeddings import LocalEmbeddings
from ragchat.search.indexer import index_chunks
from ragchat.search.opensearch_client import get_opensearch_client
from ragchat.storage.db import AsyncSessionLocal
from ragchat.storage.repository import (
    create_chunks,
    create_document,
    create_ingest_job,
    update_document_status,
    update_ingest_job,
)
from ragchat.storage.schemas import ChunkCreate, DocumentCreate, IngestJobCreate

logger = structlog.get_logger(__name__)


class IngestionState(TypedDict, total=False):
    """Shared state inside the Ingestion Graph."""

    pdf_path: str
    document_id: Optional[UUID]
    docling_doc: Optional[DoclingDocument]
    markdown_text: str
    metadata: Optional[dict]
    chunks: Optional[list[ChunkResult]]
    chunking_strategy: str  # fixed_size, recursive, structural, semantic
    ingest_job_id: Optional[UUID]
    error: Optional[str]


async def parse_node(state: IngestionState) -> IngestionState:
    """Parse PDF with Docling and create document/job in Postgres."""
    pdf_path = Path(state["pdf_path"])
    strategy = state.get("chunking_strategy", "structural")
    
    async with AsyncSessionLocal() as db:
        # Create Document entry
        doc_in = DocumentCreate(
            source_path=str(pdf_path),
            status="pending",
        )
        doc = await create_document(db, doc_in)
        
        # Create IngestJob entry
        job_in = IngestJobCreate(
            document_id=doc.id,
            stage="parse",
            status="running",
        )
        job = await create_ingest_job(db, job_in)
        await db.commit()
        
        state["document_id"] = doc.id
        state["ingest_job_id"] = job.id

    try:
        # Define output directory for parsed docling JSONs
        output_dir = Path(__file__).parent.parent.parent.parent / "data" / "parsed_docs"
        docling_doc, markdown_text, json_path = parse_pdf_to_docling(pdf_path, output_dir)
        
        async with AsyncSessionLocal() as db:
            # Update Document status and path
            await update_document_status(
                db,
                document_id=doc.id,
                status="parsed",
                docling_json_path=str(json_path),
            )
            # Update IngestJob status
            await update_ingest_job(db, job.id, status="succeeded")
            await db.commit()
            
        return {
            **state,
            "docling_doc": docling_doc,
            "markdown_text": markdown_text,
        }
    except Exception as exc:
        logger.error("ingestion_parse_failed", error=str(exc))
        async with AsyncSessionLocal() as db:
            await update_document_status(db, document_id=doc.id, status="failed")
            await update_ingest_job(db, job.id, status="failed", error=str(exc))
            await db.commit()
        return {**state, "error": str(exc)}


async def extract_metadata_node(state: IngestionState) -> IngestionState:
    """Extract metadata using heuristics or LLM fallback."""
    if "error" in state:
        return state

    doc_id = state["document_id"]
    docling_doc = state["docling_doc"]
    markdown_text = state["markdown_text"]

    async with AsyncSessionLocal() as db:
        job_in = IngestJobCreate(
            document_id=doc_id,
            stage="chunk",  # stage metadata is part of overall chunk prep
            status="running",
        )
        job = await create_ingest_job(db, job_in)
        await db.commit()

    try:
        # Extract metadata
        meta = await extract_metadata(docling_doc, markdown_text)
        
        async with AsyncSessionLocal() as db:
            # Save extracted metadata to document
            await update_document_status(
                db,
                document_id=doc_id,
                status="parsed",
                title=meta["title"],
                authors=meta["authors"],
                abstract=meta["abstract"],
            )
            await update_ingest_job(db, job.id, status="succeeded")
            await db.commit()
            
        return {**state, "metadata": meta}
    except Exception as exc:
        logger.error("ingestion_metadata_failed", error=str(exc))
        async with AsyncSessionLocal() as db:
            await update_ingest_job(db, job.id, status="failed", error=str(exc))
            await db.commit()
        return {**state, "error": str(exc)}


async def chunk_node(state: IngestionState) -> IngestionState:
    """Split document text into chunks based on strategy."""
    if "error" in state:
        return state

    doc_id = state["document_id"]
    strategy_name = state.get("chunking_strategy", "structural").lower()
    docling_doc = state["docling_doc"]
    markdown_text = state["markdown_text"]

    async with AsyncSessionLocal() as db:
        job_in = IngestJobCreate(
            document_id=doc_id,
            stage="chunk",
            status="running",
        )
        job = await create_ingest_job(db, job_in)
        await db.commit()

    try:
        # Select chunker
        if strategy_name == "fixed_size":
            chunker = FixedSizeChunker()
        elif strategy_name == "recursive":
            chunker = RecursiveMarkdownChunker()
        elif strategy_name == "semantic":
            chunker = SemanticChunker()
        else:
            chunker = StructuralChunker()

        chunks = chunker.chunk(docling_doc, markdown_text)
        
        async with AsyncSessionLocal() as db:
            await update_document_status(db, document_id=doc_id, status="chunked")
            await update_ingest_job(db, job.id, status="succeeded")
            await db.commit()
            
        return {**state, "chunks": chunks}
    except Exception as exc:
        logger.error("ingestion_chunking_failed", error=str(exc))
        async with AsyncSessionLocal() as db:
            await update_document_status(db, document_id=doc_id, status="failed")
            await update_ingest_job(db, job.id, status="failed", error=str(exc))
            await db.commit()
        return {**state, "error": str(exc)}


async def embed_and_index_node(state: IngestionState) -> IngestionState:
    """Compute embeddings, upload to OpenSearch and persist chunks in Postgres."""
    if "error" in state:
        return state

    doc_id = state["document_id"]
    chunks = state["chunks"]
    strategy_name = state.get("chunking_strategy", "structural")
    metadata = state.get("metadata") or {}

    async with AsyncSessionLocal() as db:
        job_in = IngestJobCreate(
            document_id=doc_id,
            stage="index",
            status="running",
        )
        job = await create_ingest_job(db, job_in)
        await db.commit()

    try:
        # Initialize Embeddings and OpenSearch Client
        embeddings_model = LocalEmbeddings()
        os_client = get_opensearch_client()

        # Compute embeddings for all chunks in a single batch
        chunk_texts = [c.text for c in chunks]
        embeddings = embeddings_model.embed_documents(chunk_texts)

        # Prepare payload for OpenSearch indexing and DB mapping
        os_payloads = []
        db_chunks_in = []

        for idx, chunk in enumerate(chunks):
            chunk_uuid = uuid4()
            embedding = embeddings[idx]

            payload = {
                "chunk_id": str(chunk_uuid),
                "document_id": str(doc_id),
                "chunk_index": idx,
                "text": chunk.text,
                "section_path": chunk.section_path,
                "title": metadata.get("title", ""),
                "authors": metadata.get("authors", []),
                "chunking_strategy": strategy_name,
                "embedding": embedding,
            }
            os_payloads.append(payload)

            db_chunk = ChunkCreate(
                document_id=doc_id,
                chunk_index=idx,
                section_path=chunk.section_path,
                text=chunk.text,
                token_count=chunk.token_count,
                chunking_strategy=strategy_name,
                opensearch_id=str(chunk_uuid),
            )
            db_chunks_in.append(db_chunk)

        # 1. Upload to OpenSearch
        await index_chunks(os_client, os_payloads)
        await os_client.close()

        # 2. Persist in Postgres
        async with AsyncSessionLocal() as db:
            await create_chunks(db, db_chunks_in)
            await update_document_status(db, document_id=doc_id, status="indexed")
            await update_ingest_job(db, job.id, status="succeeded")
            await db.commit()

        logger.info("ingestion_indexing_completed", doc_id=str(doc_id), chunk_count=len(chunks))
        return state
    except Exception as exc:
        logger.error("ingestion_indexing_failed", error=str(exc))
        async with AsyncSessionLocal() as db:
            await update_document_status(db, document_id=doc_id, status="failed")
            await update_ingest_job(db, job.id, status="failed", error=str(exc))
            await db.commit()
        return {**state, "error": str(exc)}


# --- Build StateGraph ---

builder = StateGraph(IngestionState)

builder.add_node("parse", parse_node)
builder.add_node("extract_metadata", extract_metadata_node)
builder.add_node("chunk", chunk_node)
builder.add_node("embed_and_index", embed_and_index_node)

builder.add_edge(START, "parse")
builder.add_edge("parse", "extract_metadata")
builder.add_edge("extract_metadata", "chunk")
builder.add_edge("chunk", "embed_and_index")
builder.add_edge("embed_and_index", END)

# Compile graph
ingestion_graph = builder.compile()
