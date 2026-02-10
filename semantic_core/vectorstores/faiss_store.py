from __future__ import annotations

from typing import List

from semantic_core.models import EmbeddedChunk, SearchQuery, SearchResult
from semantic_core.vectorstores.base import VectorStore


class FaissVectorStore(VectorStore):
    """
    In-memory FAISS vector store.

    Intended primarily for local development and testing.
    """

    def __init__(self) -> None:
        # TODO: initialize FAISS index and any auxiliary mappings
        self._index = None

    def upsert(self, items: List[EmbeddedChunk]) -> None:
        # TODO: add or update vectors in the FAISS index
        raise NotImplementedError("FaissVectorStore.upsert is not implemented yet")

    def query(self, qvec: List[float], query: SearchQuery) -> List[SearchResult]:
        # TODO: run nearest-neighbor search with FAISS and return SearchResult objects
        raise NotImplementedError("FaissVectorStore.query is not implemented yet")

    def delete_by_doc(self, doc_id: str) -> None:
        # TODO: remove all entries belonging to the given document, if supported
        raise NotImplementedError("FaissVectorStore.delete_by_doc is not implemented yet")


