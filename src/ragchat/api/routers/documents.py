"""API Router for browsing documents and their chunks."""

from pathlib import Path
from typing import Any, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from opensearchpy import AsyncOpenSearch
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from ragchat.api.deps import get_db, get_search_client
from ragchat.search.indexer import delete_all_chunks, delete_document_chunks
from ragchat.storage.repository import (
    delete_all_documents,
    delete_document,
    get_chunks_by_document,
    get_document,
    get_documents,
)
from ragchat.storage.schemas import ChunkRead, DocumentRead

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/documents", response_model=list[DocumentRead])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> list:
    """List all ingested documents in the system."""
    docs = await get_documents(db, skip=skip, limit=limit)
    return docs


@router.get("/documents/{document_id}", response_model=DocumentRead)
async def get_document_by_id(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Retrieve document metadata by UUID."""
    doc = await get_document(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    return doc


@router.get("/documents/{document_id}/chunks", response_model=list[ChunkRead])
async def get_document_chunks(
    document_id: UUID,
    strategy: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
) -> list:
    """Retrieve all stored chunks for a specific document, optionally filtered by strategy."""
    doc = await get_document(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    chunks = await get_chunks_by_document(db, document_id, strategy=strategy)
    return chunks


@router.delete("/documents/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document_by_id(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    search_client: AsyncOpenSearch = Depends(get_search_client),
) -> dict:
    """Delete a single document: its OpenSearch chunks, its Postgres rows
    (document + chunks + ingest jobs, via cascade), and its uploaded file
    on disk. Use this before re-uploading a document under a different
    chunking strategy, so stale chunks don't linger under the old strategy
    tag.
    """
    doc = await get_document(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    deleted_chunk_count = await delete_document_chunks(search_client, str(document_id))

    source_path = Path(doc.source_path) if doc.source_path else None

    await delete_document(db, document_id)
    await db.commit()

    file_removed = False
    if source_path and source_path.exists():
        try:
            source_path.unlink()
            file_removed = True
        except OSError as exc:
            logger.warning("uploaded_file_delete_failed", path=str(source_path), error=str(exc))

    logger.info(
        "document_deleted",
        document_id=str(document_id),
        opensearch_chunks_deleted=deleted_chunk_count,
        file_removed=file_removed,
    )

    return {
        "message": "Document deleted.",
        "document_id": str(document_id),
        "opensearch_chunks_deleted": deleted_chunk_count,
        "file_removed": file_removed,
    }


@router.delete("/documents", status_code=status.HTTP_200_OK)
async def delete_all_documents_endpoint(
    confirm: bool = False,
    db: AsyncSession = Depends(get_db),
    search_client: AsyncOpenSearch = Depends(get_search_client),
) -> dict:
    """Clear the entire knowledge corpus: all OpenSearch chunks, all
    Postgres document/chunk/ingest-job rows, and all uploaded files on
    disk. Requires ?confirm=true to avoid accidental full resets. Use this
    for a clean slate before re-uploading documents under a single,
    consistent chunking strategy.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This deletes the entire corpus. Pass ?confirm=true to proceed.",
        )

    docs = await get_documents(db, skip=0, limit=10000)
    source_paths = [Path(doc.source_path) for doc in docs if doc.source_path]

    deleted_chunk_count = await delete_all_chunks(search_client)
    deleted_doc_count = await delete_all_documents(db)
    await db.commit()

    files_removed = 0
    for path in source_paths:
        if path.exists():
            try:
                path.unlink()
                files_removed += 1
            except OSError as exc:
                logger.warning("uploaded_file_delete_failed", path=str(path), error=str(exc))

    logger.info(
        "corpus_cleared",
        documents_deleted=deleted_doc_count,
        opensearch_chunks_deleted=deleted_chunk_count,
        files_removed=files_removed,
    )

    return {
        "message": "Entire corpus cleared.",
        "documents_deleted": deleted_doc_count,
        "opensearch_chunks_deleted": deleted_chunk_count,
        "files_removed": files_removed,
    }
