"""Pydantic v2 schemas for API requests, responses, and internal models."""

from datetime import date, datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class IngestJobBase(BaseModel):
    stage: str
    status: str
    error: Optional[str] = None


class IngestJobCreate(IngestJobBase):
    document_id: Optional[UUID] = None


class IngestJobRead(IngestJobBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: Optional[UUID]
    started_at: datetime
    finished_at: Optional[datetime] = None


class ChunkBase(BaseModel):
    chunk_index: int
    section_path: Optional[str] = None
    text: str
    token_count: Optional[int] = None
    chunking_strategy: str
    opensearch_id: Optional[str] = None


class ChunkCreate(ChunkBase):
    document_id: UUID


class ChunkRead(ChunkBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    created_at: datetime


class DocumentBase(BaseModel):
    source_path: str
    title: Optional[str] = None
    authors: Optional[list[str]] = None
    abstract: Optional[str] = None
    publication_date: Optional[date] = None
    doc_type: Optional[str] = None
    docling_json_path: Optional[str] = None
    status: str = "pending"


class DocumentCreate(DocumentBase):
    pass


class DocumentRead(DocumentBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    ingested_at: datetime


# --- RAG Request/Response schemas ---

class Citation(BaseModel):
    document_id: UUID
    title: Optional[str] = None
    section_path: Optional[str] = None
    text: str
    chunk_index: int


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user question / query text.")
    chunking_strategy: str = Field(default="semantic", description="The chunking strategy to retrieve from.")
    provider: Optional[str] = Field(default="groq", description="LLM provider: 'groq' or 'gemini'.")
    history: list[dict] = Field(default=[], description="Conversation history list of dicts with role and content.")


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    latency_seconds: float
