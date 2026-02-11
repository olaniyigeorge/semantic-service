from __future__ import annotations

from typing import List

from semantic_core.models import NormalizedDocument, EmbeddedChunk
from semantic_core.chunking.base import Chunker
from semantic_core.embeddings.base import Embedder
from semantic_core.vectorstores.base import VectorStore
from semantic_core.metadata.base import MetadataBuilder


class IndexPipeline:
    """
    Shared indexing pipeline used by all products.
    """

    def __init__(
        self,
        *,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        metadata: MetadataBuilder,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.metadata = metadata

    def index(self, doc: NormalizedDocument) -> int:
        """
        Chunk, embed, and (optionally) upsert a single normalized document.
        """

        base_meta = self.metadata  # or: self.metadata.build_document_metadata(doc.ref)

        # --- Chunk ---
        chunks = self.chunker.chunk(doc, base_metadata=base_meta)

        # print("\n===== CHUNKS =====\n")

        # for i, c in enumerate(chunks):
        #     print(f"[Chunk {i}]")
        #     print(f"ID: {c.chunk_id}")
        #     print(f"Doc ID: {c.doc_id}")
        #     print(f"Text length: {len(c.text)}")
        #     print(f"Text preview: {c.text[:40]}")
        #     print(f"Metadata: {c.metadata}")
        #     print(f"Page: {c.page_number}")
        #     print(f"Chars: {c.start_char} -> {c.end_char}")
        #     print("-" * 60)

        # --- Embed ---
        vectors = self.embedder.embed_texts([c.text for c in chunks])

        # --- Pair chunks + vectors ---
        embedded: List[EmbeddedChunk] = [
            EmbeddedChunk(chunk=c, vector=v)
            for c, v in zip(chunks, vectors)
        ]

        # print("\n===== EMBEDDED CHUNKS =====\n")

        # for i, e in enumerate(embedded):
        #     vec = e.vector.values if hasattr(e.vector, "values") else e.vector

        #     print(f"[Embedded {i}]")
        #     print(f"Chunk ID: {e.chunk.chunk_id}")
        #     print(f"First 8 Vector dims: {vec[:8]}")
        #     print("-" * 60)

        # # --- Store (disabled for debugging) ---
        self.store.upsert(embedded)

        return len(embedded)


