"""Script to ingest local PDF documents into Qdrant Cloud Vector Index using SemanticChunker."""

import sys
import os
from pathlib import Path
import structlog
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Load .env file explicitly
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragchat.config import settings
from ragchat.search.embeddings import LocalEmbeddings
from ragchat.chunking.semantic import SemanticChunker

logger = structlog.get_logger(__name__)


def ingest_documents_to_qdrant():
    """Ingests all PDFs from data/ and src/data/uploads into Qdrant Cloud using Semantic Chunker."""
    qdrant_url = os.environ.get("QDRANT_URL") or getattr(settings, "qdrant_url", None)
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or getattr(settings, "qdrant_api_key", None)

    if not qdrant_url:
        print("ERROR: Please set QDRANT_URL in your .env file!")
        return

    print(f"Connecting to Qdrant Cloud: {qdrant_url}...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    collection_name = "chunks"
    embedder = LocalEmbeddings()
    chunker = SemanticChunker()

    # 1. Create collection if it doesn't exist
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        print(f"Creating Qdrant collection '{collection_name}' (384 dim)...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        )

    # 2. Search PDF Files in data/ and src/data/uploads
    search_dirs = [Path("data"), Path("src/data/uploads")]
    pdf_files = []
    for d in search_dirs:
        if d.exists():
            pdf_files.extend(list(d.glob("*.pdf")))

    print(f"Found {len(pdf_files)} PDF files across search directories...")

    for pdf in pdf_files:
        print(f"Processing: {pdf.name} using SemanticChunker...")
        try:
            # Read text from file via Docling converter
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(str(pdf))
            text_content = result.document.export_to_markdown()

            chunks = chunker.chunk(result.document, text_content)
            print(f"  Generated {len(chunks)} semantic chunks for {pdf.name}.")

            if not chunks:
                continue

            texts = [c.text for c in chunks]
            print("  Generating 384d BGE embeddings...")
            vectors = embedder.embed_documents(texts)

            points = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                point_id = hash(f"{pdf.name}_{i}_{chunk.text[:50]}") & 0x7FFFFFFFFFFFFFFF
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "doc_title": pdf.name,
                        "strategy": "semantic",
                        "metadata": {"section_path": chunk.section_path, "token_count": chunk.token_count},
                    }
                ))

            # Upload to Qdrant Cloud
            client.upsert(collection_name=collection_name, points=points)
            print(f"  ✅ Successfully uploaded {len(points)} semantic vectors for {pdf.name} to Qdrant Cloud!")

        except Exception as exc:
            print(f"  ❌ Error processing {pdf.name}: {exc}")

    print("\n🎉 ALL DOCUMENTS INGESTED INTO QDRANT CLOUD SUCCESSFULLY!")


if __name__ == "__main__":
    ingest_documents_to_qdrant()
