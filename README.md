# Semantic Service

A standalone **semantic infrastructure service** for document ingestion, chunking, embedding, storage, and search.

This project provides a modular backend system designed to power **semantic search, Retrieval-Augmented Generation (RAG), and recommendation systems**. It abstracts the full semantic pipeline into reusable components that can be embedded into multiple products and services.

---

## Overview

Semantic Service is built as an **independent microservice** responsible for transforming raw documents into searchable semantic representations.

It supports:

* Document ingestion via API
* Multiple chunking strategies per document type
* Pluggable embedding models (Gemini by default)
* Vector storage backends (pgvector, Qdrant, FAISS)
* High-level indexing and search pipelines
* Metadata enrichment for domain-specific use cases

The architecture is adapter-driven, allowing teams to extend chunking, embedding, and storage strategies without modifying core logic.

---

## Goals

This service is designed to evolve into a full **semantic engineering platform** that powers:

* Retrieval-Augmented Generation (RAG)
* Semantic search systems
* Recommendation engines
* Domain-specific knowledge indexing
* Cross-product semantic infrastructure

The long-term vision is to provide a reusable semantic layer that multiple applications can share.

---

## Architecture

The system is divided into two primary layers:

### 1. `semantic_core`

Contains reusable domain logic and pipelines:

* Data models
* Document normalization
* Chunking strategies
* Embedding adapters
* Vector store abstractions
* Index and search pipelines
* Metadata builders

This layer is framework-agnostic and can be reused outside FastAPI.

### 2. `semantic_api`

Provides a FastAPI interface for:

* Document ingestion endpoints
* Semantic search endpoints
* Dependency injection wiring

This layer exposes the semantic functionality as a web service.

---

## Project Structure

```
semantic-service/
  semantic_core/
    models.py
    ingest/
    chunking/
    embeddings/
      base.py
      gemini.py
    vectorstores/
      base.py
      pgvector_store.py
      qdrant_store.py
      faiss_store.py
    pipeline/
      indexer.py
      searcher.py
    metadata/
      coopwise.py
      winnov8.py

  semantic_api/
    main.py
    routes/
      documents.py
      search.py
    deps.py

  docker-compose.yml
  Dockerfile
  pyproject.toml
```

---

## Features

### Modular Chunking

Different document types can use specialized chunking strategies:

* Text documents
* JSON documents
* HTML documents
* PDF documents
* Extensible adapters for custom formats

### Pluggable Embeddings

The system uses **Google Gemini embeddings** by default, but the embedding interface allows swapping in:

* OpenAI
* Local embedding models
* Custom embedding services

### Multiple Vector Stores

Supported storage backends:

* PostgreSQL + pgvector
* Qdrant
* FAISS (development/testing)

Each backend implements a shared vector store interface.

### Pipelines

Two high-level pipelines orchestrate the system:

* **IndexPipeline**: chunk → embed → store
* **SearchPipeline**: embed query → retrieve results

---

## Installation

### Using Docker

```bash
docker-compose up --build
```

### Local Development

```bash
pip install -e .
uvicorn semantic_api.main:app --reload
```

---

## API Endpoints

### Ingest Document

```
POST /documents/ingest
```

Accepts raw text or file uploads and indexes them into the semantic store.

### Search

```
POST /search
```

Runs semantic similarity search over indexed documents.

---

## Extending the System

### Add a New Embedding Model

Implement the embedding interface:

```
semantic_core/embeddings/base.py
```

Then register your adapter.

### Add a New Chunker

Create a chunker strategy in:

```
semantic_core/chunking/
```

And route it via the chunking dispatcher.

### Add a New Vector Store

Implement:

```
semantic_core/vectorstores/base.py
```

---

## Use Cases

This service is intended to power:

* Knowledge retrieval systems
* Document search engines
* RAG pipelines
* Recommendation platforms
* Domain-specific semantic indexing

---

## Future Roadmap

* Incremental indexing
* Advanced metadata filtering
* Reranking pipelines
* Distributed indexing
* Streaming ingestion
* Monitoring and observability
* Multi-tenant semantic infrastructure

---

## License

Internal project. Licensing to be defined.

---

## Contributing

Contributions should maintain modularity and follow adapter-based design principles. Each new feature should be extensible and pipeline-compatible.

---

## Summary

Semantic Service is a reusable semantic engine that transforms documents into searchable knowledge. It separates ingestion, chunking, embedding, and storage into clean interfaces, enabling scalable semantic systems across products.
