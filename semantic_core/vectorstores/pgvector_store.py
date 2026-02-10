from __future__ import annotations

from typing import Iterable, List, Sequence

from semantic_core.models import Chunk, SearchQuery, SearchResult
from semantic_core.vectorstores.base import VectorStore


class PgVectorStore(VectorStore):
    """
    PostgreSQL + pgvector based vector store.

    Implement connection management and SQL schema in a later iteration.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    async def index_chunks(self, chunks: Iterable[Chunk]) -> None:
        # TODO: implement bulk upsert into pgvector-backed table
        raise NotImplementedError("PgVectorStore.index_chunks is not implemented yet")

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        # TODO: implement similarity search using pgvector operators
        raise NotImplementedError("PgVectorStore.search is not implemented yet")

    async def delete_by_document_ids(self, document_ids: Sequence[str]) -> None:
        # TODO: implement deletion of rows belonging to given documents
        raise NotImplementedError(
            "PgVectorStore.delete_by_document_ids is not implemented yet"
        )


