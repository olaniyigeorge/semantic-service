from __future__ import annotations

from typing import List

from semantic_core.embeddings.base import Embedder
from semantic_core.vectorstores.base import VectorStore
from semantic_core.models import SearchQuery, SearchResult


class SearchPipeline:
    """
    High-level pipeline that embeds queries and delegates to a VectorStore.
    """

    def __init__(self, *, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Embed the query text and execute a vector search against the store.
        """
        qvec = self._embedder.embed_texts([query.query])[0]
        return self._store.query(qvec, query)


