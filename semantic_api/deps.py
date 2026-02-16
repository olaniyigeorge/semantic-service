from __future__ import annotations
import os
import logging
from functools import lru_cache

from typing import Literal
from fastapi import Depends
from qdrant_client import QdrantClient

from semantic_core.embeddings.gemini import GeminiEmbeddings
from semantic_core.ingest.normalizer import DocumentNormalizer
from semantic_core.pipelines.searcher import SearchPipeline
from semantic_core.pipelines.indexer import IndexPipeline
from semantic_core.vectorstores.base import VectorStore
from semantic_core.vectorstores.faiss_store import FaissVectorStore
from semantic_core.chunking.base import Chunker, build_default_chunker
from semantic_core.vectorstores.pgvector_store import PgVectorStore
from semantic_core.vectorstores.qdrant_store import QDrantVectorStore

logger = logging.getLogger(__name__)


@lru_cache
def get_embeddings() -> GeminiEmbeddings:
    # TODO: Switch case of embedding models. Gemini by default
    return GeminiEmbeddings()


@lru_cache
def get_chunker() -> Chunker:
    return Chunker()


VectorStoreType = Literal["qdrant", "faiss", "pgvector"]


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """
    Get or create a Qdrant client instance.

    Configuration via environment variables:
    - QDRANT_HOST: Qdrant server host (default: localhost)
    - QDRANT_PORT: Qdrant server port (default: 6333)
    - QDRANT_API_KEY: Optional API key for authentication
    - QDRANT_CLUSTER_URL: Full URL (alternative to host/port)
    - QDRANT_IN_MEMORY: Use in-memory mode for testing (default: false)
    """
    use_memory = os.getenv("QDRANT_IN_MEMORY", "false").lower() == "true"

    if use_memory:
        logger.info("Using in-memory Qdrant client")
        return QdrantClient(":memory:")

    # Check for full URL first
    qdrant_url = os.getenv("QDRANT_CLUSTER_URL")
    if qdrant_url:
        api_key = os.getenv("QDRANT_API_KEY")
        logger.info(f"\nConnecting to Qdrant at URL: {qdrant_url}\n")
        return QdrantClient(url=qdrant_url, api_key=api_key)

    # Fall back to host/port configuration
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    api_key = os.getenv("QDRANT_API_KEY")

    logger.info(f"Connecting to Qdrant at {host}:{port}")
    return QdrantClient(host=host, port=port, api_key=api_key, timeout=60)


def _create_qdrant_store() -> QDrantVectorStore:
    """Create and return a Qdrant vector store instance."""
    client = get_qdrant_client()
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "semantic_chunks")
    embeddings = get_embeddings()

    logger.info(f"Creating Qdrant vector store with collection: {collection_name}")
    return QDrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding_model=embeddings,
        vector_size=None,
    )


@lru_cache
def get_vector_store() -> VectorStore:
    """
    Get the configured vector store instance.

    Supports multiple vector store backends:
    - qdrant (default): Qdrant vector database
    - faiss: FAISS in-memory/file-based store
    - pgvector: PostgreSQL with pgvector extension

    Configuration via environment variable:
    - VECTOR_STORE_TYPE: Type of vector store (default: qdrant)

    Environment variables by store type:

    Qdrant:
    - QDRANT_URL or QDRANT_HOST/QDRANT_PORT
    - QDRANT_API_KEY (optional)
    - QDRANT_COLLECTION_NAME (default: semantic_chunks)
    - QDRANT_IN_MEMORY (default: false)

    FAISS:
    - No additional configuration needed

    PgVector:
    - PGVECTOR_CONNECTION_STRING or PGVECTOR_HOST/PORT/DATABASE/USER/PASSWORD
    - PGVECTOR_TABLE_NAME (default: embeddings)

    Returns:
        VectorStore: Configured vector store instance

    Raises:
        ValueError: If an unsupported vector store type is specified
    """
    store_type = os.getenv("VECTOR_STORE_TYPE", "qdrant").lower()

    logger.info(f"Initializing vector store: {store_type}")

    if store_type == "qdrant":
        return _create_qdrant_store()
    elif store_type == "faiss":
        return _create_faiss_store()
    elif store_type == "pgvector":
        return _create_pgvector_store()
    else:
        raise ValueError(
            f"Unsupported vector store type: {store_type}. "
            f"Supported types: qdrant, faiss, pgvector"
        )


def _create_pgvector_store() -> PgVectorStore:
    """
    Create and return a PgVector store instance.

    Configuration via environment variables:
    - PGVECTOR_CONNECTION_STRING or individual components:
    - PGVECTOR_HOST: PostgreSQL host (default: localhost)
    - PGVECTOR_PORT: PostgreSQL port (default: 5432)
    - PGVECTOR_DATABASE: Database name (default: vector_db)
    - PGVECTOR_USER: Database user (default: postgres)
    - PGVECTOR_PASSWORD: Database password
    - PGVECTOR_TABLE_NAME: Table name for vectors (default: embeddings)
    """
    # Try full connection string first
    connection_string = os.getenv("PGVECTOR_CONNECTION_STRING")

    if not connection_string:
        # Build from individual components
        host = os.getenv("PGVECTOR_HOST", "localhost")
        port = os.getenv("PGVECTOR_PORT", "5432")
        database = os.getenv("PGVECTOR_DATABASE", "vector_db")
        user = os.getenv("PGVECTOR_USER", "postgres")
        password = os.getenv("PGVECTOR_PASSWORD", "")

        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    table_name = os.getenv("PGVECTOR_TABLE_NAME", "embeddings")
    embeddings = get_embeddings()

    logger.info(f"Creating PgVector store with table: {table_name}")
    return PgVectorStore(
        connection_string=connection_string,
        table_name=table_name,
        embedding_model=embeddings,
    )


def _create_faiss_store() -> FaissVectorStore:
    """Create and return a FAISS vector store instance."""
    logger.info("Creating FAISS vector store")
    return FaissVectorStore()


# Helper function to clear the cache and reinitialize
def reset_vector_store() -> None:
    """
    Clear the cached vector store instance.
    Useful when you need to reinitialize with different configuration.
    """
    get_vector_store.cache_clear()
    get_qdrant_client.cache_clear()
    logger.info("Vector store cache cleared")


def get_document_normalizer() -> DocumentNormalizer:
    return DocumentNormalizer()


# ------- PIPELINES -------


def get_search_pipeline() -> SearchPipeline:
    embedder = get_embeddings()
    vector_store = get_vector_store()
    return SearchPipeline(embedder=embedder, store=vector_store)


def get_index_pipeline() -> IndexPipeline:
    """
    Build the default indexing pipeline with the configured vector store.

    The vector store type is determined by the VECTOR_STORE_TYPE environment
    variable (default: qdrant).
    """
    chunker = build_default_chunker()
    embedder = get_embeddings()
    store = get_vector_store()
    metadata = {"source": "semantic_api"}

    logger.info(f"\nIndex pipeline created with {type(store).__name__}\n")

    return IndexPipeline(
        chunker=chunker, embedder=embedder, store=store, metadata=metadata
    )
