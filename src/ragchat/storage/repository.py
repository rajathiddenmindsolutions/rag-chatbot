"""Database Repository module for CRUD operations on documents, chunks, and ingestion jobs."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ragchat.storage.models import Chunk, Document, IngestJob
from ragchat.storage.schemas import ChunkCreate, DocumentCreate, IngestJobCreate


async def get_document(db: AsyncSession, document_id: UUID) -> Optional[Document]:
    """Retrieve a document by its UUID."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    return result.scalars().first()


async def get_document_by_path(db: AsyncSession, source_path: str) -> Optional[Document]:
    """Retrieve a document by its file source path."""
    result = await db.execute(select(Document).where(Document.source_path == source_path))
    return result.scalars().first()


async def get_documents(db: AsyncSession, skip: int = 0, limit: int = 100) -> list[Document]:
    """Retrieve multiple documents with pagination."""
    result = await db.execute(select(Document).offset(skip).limit(limit).order_by(Document.ingested_at.desc()))
    return list(result.scalars().all())


async def create_document(db: AsyncSession, doc_in: DocumentCreate) -> Document:
    """Create a new document entry in the database."""
    db_doc = Document(
        source_path=doc_in.source_path,
        title=doc_in.title,
        authors=doc_in.authors,
        abstract=doc_in.abstract,
        publication_date=doc_in.publication_date,
        doc_type=doc_in.doc_type,
        docling_json_path=doc_in.docling_json_path,
        status=doc_in.status,
    )
    db.add(db_doc)
    await db.flush()  # Populates db_doc.id without committing
    return db_doc


async def update_document_status(
    db: AsyncSession,
    document_id: UUID,
    status: str,
    title: Optional[str] = None,
    authors: Optional[list[str]] = None,
    abstract: Optional[str] = None,
    docling_json_path: Optional[str] = None,
) -> Optional[Document]:
    """Update status and metadata fields of a document."""
    doc = await get_document(db, document_id)
    if not doc:
        return None
    
    doc.status = status
    if title is not None:
        doc.title = title
    if authors is not None:
        doc.authors = authors
    if abstract is not None:
        doc.abstract = abstract
    if docling_json_path is not None:
        doc.docling_json_path = docling_json_path

    db.add(doc)
    await db.flush()
    return doc


async def create_chunks(db: AsyncSession, chunks_in: list[ChunkCreate]) -> list[Chunk]:
    """Bulk create chunks for a document."""
    db_chunks = []
    for chunk_in in chunks_in:
        db_chunk = Chunk(
            document_id=chunk_in.document_id,
            chunk_index=chunk_in.chunk_index,
            section_path=chunk_in.section_path,
            text=chunk_in.text,
            token_count=chunk_in.token_count,
            chunking_strategy=chunk_in.chunking_strategy,
            opensearch_id=chunk_in.opensearch_id,
        )
        db.add(db_chunk)
        db_chunks.append(db_chunk)
    await db.flush()
    return db_chunks


async def get_chunks_by_document(
    db: AsyncSession, document_id: UUID, strategy: Optional[str] = None
) -> list[Chunk]:
    """Retrieve all chunks belonging to a document, optionally filtered by strategy."""
    stmt = select(Chunk).where(Chunk.document_id == document_id)
    if strategy:
        stmt = stmt.where(Chunk.chunking_strategy == strategy)
    stmt = stmt.order_by(Chunk.chunk_index.asc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def delete_document(db: AsyncSession, document_id: UUID) -> Optional[Document]:
    """Delete a document by UUID. Cascades to its chunks and ingest jobs
    (both at the ORM level via cascade='all, delete-orphan' and at the DB
    level via ondelete=CASCADE on the foreign keys). Returns the deleted
    Document object (detached) if it existed, else None. Caller is
    responsible for also removing the document's chunks from OpenSearch
    and its source file from disk — this only handles Postgres.
    """
    doc = await get_document(db, document_id)
    if not doc:
        return None
    await db.delete(doc)
    await db.flush()
    return doc


async def delete_all_documents(db: AsyncSession) -> int:
    """Delete every document (and, via cascade, all chunks/ingest jobs).
    Returns the number of documents deleted. Caller is responsible for also
    clearing the OpenSearch index and any uploaded files on disk.
    """
    result = await db.execute(select(Document))
    docs = list(result.scalars().all())
    count = len(docs)
    for doc in docs:
        await db.delete(doc)
    await db.flush()
    return count


async def create_ingest_job(db: AsyncSession, job_in: IngestJobCreate) -> IngestJob:
    """Create a tracking ingestion job."""
    db_job = IngestJob(
        document_id=job_in.document_id,
        stage=job_in.stage,
        status=job_in.status,
        error=job_in.error,
        started_at=datetime.utcnow(),
    )
    db.add(db_job)
    await db.flush()
    return db_job


async def update_ingest_job(
    db: AsyncSession, job_id: UUID, status: str, error: Optional[str] = None
) -> Optional[IngestJob]:
    """Update ingestion job status and set finished timestamp."""
    result = await db.execute(select(IngestJob).where(IngestJob.id == job_id))
    job = result.scalars().first()
    if not job:
        return None
    
    job.status = status
    if error is not None:
        job.error = error
    job.finished_at = datetime.utcnow()
    
    db.add(job)
    await db.flush()
    return job
