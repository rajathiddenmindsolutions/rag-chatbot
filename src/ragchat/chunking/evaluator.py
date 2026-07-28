"""Harness to evaluate and compare different chunking strategies."""

import numpy as np
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter

from ragchat.chunking.fixed_size import FixedSizeChunker
from ragchat.chunking.recursive import RecursiveMarkdownChunker
from ragchat.chunking.structural import StructuralChunker
from ragchat.chunking.semantic import SemanticChunker


class ChunkingEvaluator:
    """Evaluates chunking strategies on a test document with queries."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def evaluate_strategy(
        self,
        strategy_name: str,
        chunker,
        docling_doc,
        markdown_text: str,
        queries_with_keywords: list[dict],
    ) -> dict:
        """Evaluate a single chunking strategy."""
        # 1. Chunk document
        chunks = chunker.chunk(docling_doc, markdown_text)
        if not chunks:
            return {
                "strategy": strategy_name,
                "chunk_count": 0,
                "avg_token_count": 0.0,
                "hit_rate_at_3": 0.0,
                "mrr_at_3": 0.0,
            }

        chunk_texts = [c.text for c in chunks]
        avg_tokens = np.mean([c.token_count for c in chunks])

        # 2. Embed chunks
        chunk_embeddings = self.model.encode(chunk_texts, convert_to_numpy=True)

        hits = 0
        rr_sum = 0.0
        num_queries = len(queries_with_keywords)

        # 3. For each query, retrieve top 3 by cosine similarity
        for item in queries_with_keywords:
            query = item["query"]
            keywords = item["keywords"]

            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # Compute similarities
            similarities = []
            for emb in chunk_embeddings:
                norm1 = np.linalg.norm(query_embedding)
                norm2 = np.linalg.norm(emb)
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(query_embedding, emb) / (norm1 * norm2)
                else:
                    sim = 0.0
                similarities.append(sim)

            # Sort and get top 3 indices
            top_indices = np.argsort(similarities)[::-1][:3]
            
            # Check if any top chunk matches any of the keywords
            found = False
            for rank, idx in enumerate(top_indices, 1):
                chunk_text = chunk_texts[idx].lower()
                if any(kw.lower() in chunk_text for kw in keywords):
                    if not found:
                        hits += 1
                        rr_sum += 1.0 / rank
                        found = True

        return {
            "strategy": strategy_name,
            "chunk_count": len(chunks),
            "avg_token_count": float(avg_tokens),
            "hit_rate_at_3": hits / num_queries if num_queries > 0 else 0.0,
            "mrr_at_3": rr_sum / num_queries if num_queries > 0 else 0.0,
        }
