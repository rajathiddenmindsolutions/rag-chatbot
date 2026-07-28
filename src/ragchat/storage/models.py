"""SQLAlchemy ORM models representing Postgres schema."""

import uuid
from datetime import datetime
from sqlalchemy import ARRAY, Column, Date, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_path = Column(Text, nullable=False)
    title = Column(Text, nullable=True)
    authors = Column(ARRAY(Text), nullable=True)
    abstract = Column(Text, nullable=True)
    publication_date = Column(Date, nullable=True)
    doc_type = Column(String, nullable=True)
    docling_json_path = Column(Text, nullable=True)
    ingested_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    status = Column(String, default="pending")

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    ingest_jobs = relationship("IngestJob", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    section_path = Column(Text, nullable=True)
    text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    chunking_strategy = Column(String, nullable=False)
    opensearch_id = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", "chunking_strategy", name="chunks_document_id_chunk_index_chunking_strategy_key"),
    )


class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True)
    stage = Column(String, nullable=False)  # parse, chunk, embed, index
    status = Column(String, nullable=False)  # running, succeeded, failed
    error = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    document = relationship("Document", back_populates="ingest_jobs")
