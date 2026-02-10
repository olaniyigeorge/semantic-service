from __future__ import annotations

from typing import Protocol, List, Optional, Dict, Any

from semantic_core.models import EmbeddedChunk, SearchQuery, SearchResult


class VectorStore(Protocol):
    """
    Abstract interface for vector search backends.

    Concrete implementations can wrap FAISS, pgvector, Qdrant, etc.
    """

    def upsert(self, items: List[EmbeddedChunk]) -> None:
        """
        Insert or update embedded chunks in the underlying store.
        """
        ...

    def delete_by_doc(self, doc_id: str) -> None:
        """
        Remove all chunks belonging to the given logical document.
        """
        ...

    def query(self, qvec: List[float], query: SearchQuery) -> List[SearchResult]:
        """
        Run a vector similarity search using the provided query vector and
        additional filtering / ranking parameters.
        """
        ...

