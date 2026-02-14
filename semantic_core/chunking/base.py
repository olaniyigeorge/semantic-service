from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter

from semantic_core.models import NormalizedDocument
from semantic_core.models import Chunk  # adjust import if Chunk is elsewhere


class Chunker(Protocol):
    """
    Strategy interface for turning normalized documents into semantic chunks.
    """

    def chunk(
        self,
        doc: NormalizedDocument,
        *,
        base_metadata: Dict[str, Any],
    ) -> List[Chunk]: ...


def _chunk_id(*parts: str) -> str:
    """Stable chunk id from deterministic parts."""
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _merge_meta(base_metadata: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure base metadata never gets mutated.
    return {**base_metadata, **extra}


@dataclass
class TextChunker:
    """
    General-purpose chunker for txt/raw/docx/csv-ish text using LangChain's recursive splitter.
    """

    chunk_size: int = 1200
    chunk_overlap: int = 150

    def __post_init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(
        self, doc: NormalizedDocument, *, base_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        pieces = self._splitter.split_text(doc.text or "")
        out: List[Chunk] = []

        # We don't have accurate char offsets from LangChain splitter without extra work,
        # but we can still provide chunk_index and stable ids.
        for i, text in enumerate(pieces):
            out.append(
                Chunk(
                    chunk_id=_chunk_id(doc.ref.checksum, f"t{i}"),
                    doc_id=doc.ref.doc_id,
                    text=text,
                    metadata=_merge_meta(
                        base_metadata,
                        {"chunk_index": i, "doc_type": doc.ref.doc_type},
                    ),
                    # start_char/end_char optional (unknown here)
                )
            )
        return out


def _flatten_json(obj: Any, prefix: str = "") -> List[tuple[str, str]]:
    """
    Flatten JSON into path:value lines.
    Example: {"a":{"b":2}} -> [("a.b","2")]
    """
    items: List[tuple[str, str]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            items.extend(_flatten_json(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            path = f"{prefix}[{i}]"
            items.extend(_flatten_json(v, path))
    else:
        if prefix:
            items.append((prefix, "" if obj is None else str(obj)))
    return items


@dataclass
class JsonChunker:
    """
    Chunk JSON by flattening to searchable text then using TextChunker.
    This works well for Coopwise records (contributions, cover letters, requests).
    """

    text_chunker: TextChunker

    def chunk(
        self, doc: NormalizedDocument, *, base_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        raw = doc.text or ""
        try:
            obj = json.loads(raw)
        except Exception:
            # If not valid JSON, treat as plain text.
            return self.text_chunker.chunk(doc, base_metadata=base_metadata)

        lines: List[str] = []
        for path, value in _flatten_json(obj):
            if value.strip():
                lines.append(f"{path}: {value}")

        flattened = "\n".join(lines).strip()
        proxy = NormalizedDocument(ref=doc.ref, text=flattened, pages=None)

        chunks = self.text_chunker.chunk(proxy, base_metadata=base_metadata)
        return [
            Chunk(
                chunk_id=_chunk_id(doc.ref.checksum, f"j{i}"),
                doc_id=c.doc_id,
                text=c.text,
                metadata=_merge_meta(c.metadata, {"json_flattened": True}),
                page_number=c.page_number,
                start_char=c.start_char,
                end_char=c.end_char,
            )
            for i, c in enumerate(chunks)
        ]


@dataclass
class HtmlChunker:
    """
    Chunk HTML by extracting visible text, preserving headings, then TextChunker.
    """

    text_chunker: TextChunker

    def chunk(
        self, doc: NormalizedDocument, *, base_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(doc.text or "", "html.parser")

        # Remove common non-content tags
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        blocks: List[str] = []
        for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
            t = el.get_text(" ", strip=True)
            if not t:
                continue
            if el.name in ("h1", "h2", "h3"):
                blocks.append(f"\n\n## {t}\n")
            else:
                blocks.append(t)

        extracted = "\n".join(blocks).strip()
        proxy = NormalizedDocument(ref=doc.ref, text=extracted, pages=None)

        chunks = self.text_chunker.chunk(proxy, base_metadata=base_metadata)
        return [
            Chunk(
                chunk_id=_chunk_id(doc.ref.checksum, f"h{i}"),
                doc_id=c.doc_id,
                text=c.text,
                metadata=_merge_meta(c.metadata, {"html_extracted": True}),
                page_number=c.page_number,
                start_char=c.start_char,
                end_char=c.end_char,
            )
            for i, c in enumerate(chunks)
        ]


@dataclass
class PdfChunker:
    """
    Chunk PDFs page-aware using doc.pages when available.
    Falls back to TextChunker on doc.text.
    """

    text_chunker: TextChunker

    def chunk(
        self, doc: NormalizedDocument, *, base_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        if not doc.pages:
            # fallback: big string chunking
            chunks = self.text_chunker.chunk(doc, base_metadata=base_metadata)
            return [
                Chunk(
                    chunk_id=_chunk_id(doc.ref.checksum, f"p0_{i}"),
                    doc_id=c.doc_id,
                    text=c.text,
                    metadata=_merge_meta(c.metadata, {"pdf_page_aware": False}),
                    page_number=c.page_number,
                    start_char=c.start_char,
                    end_char=c.end_char,
                )
                for i, c in enumerate(chunks)
            ]

        out: List[Chunk] = []
        for page_idx, page_text in enumerate(doc.pages, start=1):
            if not (page_text or "").strip():
                continue
            proxy = NormalizedDocument(ref=doc.ref, text=page_text, pages=None)
            page_chunks = self.text_chunker.chunk(
                proxy,
                base_metadata=_merge_meta(base_metadata, {"page_number": page_idx}),
            )
            for i, c in enumerate(page_chunks):
                out.append(
                    Chunk(
                        chunk_id=_chunk_id(doc.ref.checksum, f"p{page_idx}_{i}"),
                        doc_id=doc.ref.doc_id,
                        text=c.text,
                        metadata=_merge_meta(
                            c.metadata,
                            {"pdf_page_aware": True, "page_number": page_idx},
                        ),
                        page_number=page_idx,
                        start_char=None,
                        end_char=None,
                    )
                )
        return out


@dataclass
class RouterChunker:
    """
    Dispatch to a chunker based on doc.ref.doc_type.
    """

    by_doc_type: Mapping[str, Chunker]
    fallback: Chunker

    def chunk(
        self, doc: NormalizedDocument, *, base_metadata: Dict[str, Any]
    ) -> List[Chunk]:
        impl = self.by_doc_type.get(doc.ref.doc_type, self.fallback)
        return impl.chunk(doc, base_metadata=base_metadata)


def build_default_chunker() -> RouterChunker:
    """
    A sensible default router for your pipeline.
    """
    text = TextChunker(chunk_size=1200, chunk_overlap=150)
    return RouterChunker(
        by_doc_type={
            "txt": text,
            "raw": text,
            "docx": text,  # later you can make this structure-aware
            "csv": text,  # later you can make this row-aware
            "json": JsonChunker(text_chunker=text),
            "html": HtmlChunker(text_chunker=text),
            "pdf": PdfChunker(text_chunker=text),
        },
        fallback=text,
    )
