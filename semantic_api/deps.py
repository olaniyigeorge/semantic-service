from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from semantic_core.embeddings.gemini import GeminiEmbeddings
from semantic_core.pipeline.searcher import SearchPipeline
from semantic_core.pipeline.indexer import IndexPipeline
from semantic_core.vectorstores.faiss_store import FaissVectorStore
from semantic_core.chunking.base import Chunker, build_default_chunker


@lru_cache
def get_embeddings() -> GeminiEmbeddings:
    # Switch case of embedding models. Gemini by default
    return GeminiEmbeddings()


@lru_cache
def get_vector_store() -> FaissVectorStore:
    # TODO: swap to PgVectorStore in production
    return FaissVectorStore()

@lru_cache
def get_chunker() -> Chunker:
    # TODO: swap to PgVectorStore in production
    return Chunker()




def get_search_pipeline(
    vector_store: FaissVectorStore = Depends(get_vector_store),
) -> SearchPipeline:
    embedder = get_embeddings()
    return SearchPipeline(embedder=embedder, store=vector_store)

def get_index_pipeline() -> IndexPipeline:
    chunker = build_default_chunker()
    embedder = get_embeddings()
    store = get_vector_store()
    metadata = {"source": "semantic_api"}
    return IndexPipeline(chunker=chunker, embedder=embedder, store=store, metadata=metadata)
