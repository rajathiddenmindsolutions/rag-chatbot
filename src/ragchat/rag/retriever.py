"""Custom LangChain Retriever wrapping OpenSearch hybrid search."""

from typing import Any, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from opensearchpy import AsyncOpenSearch

from ragchat.search.embeddings import LocalEmbeddings
from ragchat.search.hybrid_search import hybrid_search


class OpenSearchHybridRetriever(BaseRetriever):
    """Custom retriever performing BM25 + vector search + RRF fusion on OpenSearch."""

    client: Any  # AsyncOpenSearch instance
    embeddings_model: Any  # LocalEmbeddings instance
    chunking_strategy: Optional[str] = None
    top_k: int = 6
    min_score: float = 0.015

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> list[Document]:
        """Synchronous retrieval fallback."""
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # In an async context, run synchronously via executor or run_coroutine_threadsafe
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._aget_relevant_documents(query)
                )
                return future.result()
        else:
            return asyncio.run(self._aget_relevant_documents(query))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None
    ) -> list[Document]:
        """Asynchronously fetch documents using hybrid search."""
        query_vector = self.embeddings_model.embed_query(query)
        hits = await hybrid_search(
            client=self.client,
            query_text=query,
            query_vector=query_vector,
            strategy=self.chunking_strategy,
            top_k=self.top_k,
            min_score=self.min_score,
        )
        
        docs = []
        for h in hits:
            docs.append(
                Document(
                    page_content=h["text"],
                    metadata={
                        "document_id": h["document_id"],
                        "chunk_id": h["chunk_id"],
                        "chunk_index": h["chunk_index"],
                        "section_path": h["section_path"],
                        "title": h["title"],
                        "authors": h["authors"],
                        "score": h["score"],
                    },
                )
            )
        return docs
