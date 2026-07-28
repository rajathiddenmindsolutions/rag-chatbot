-- Runs automatically on first container start (docker-entrypoint-initdb.d).
-- Re-running requires a fresh volume (or wrap in DROP TABLE IF EXISTS for dev resets).

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- for gen_random_uuid()

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_path TEXT NOT NULL,
    title TEXT,
    authors TEXT[],
    abstract TEXT,
    publication_date DATE,
    doc_type TEXT,
    docling_json_path TEXT,
    ingested_at TIMESTAMPTZ DEFAULT now(),
    status TEXT DEFAULT 'pending'
        CHECK (status IN ('pending', 'parsed', 'chunked', 'indexed', 'failed'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    section_path TEXT,
    text TEXT NOT NULL,
    token_count INT,
    chunking_strategy TEXT,
    opensearch_id TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (document_id, chunk_index, chunking_strategy)
);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    stage TEXT CHECK (stage IN ('parse', 'chunk', 'embed', 'index')),
    status TEXT CHECK (status IN ('running', 'succeeded', 'failed')),
    error TEXT,
    started_at TIMESTAMPTZ DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_document_id ON ingest_jobs(document_id);
