from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

DocType = Literal[
    "policy",
    "education",
    "report",
    "platform_doc",
    "resume",
    "cover_letter",
    "project_doc",
    "collaboration_request",
    "application",
    "other",
]


@dataclass(frozen=True)
class DocumentRef:
    """
    Stable reference information about a logical document.
    """

    doc_id: str
    source: str  # filename/url/path/etc. for traceability
    doc_type: DocType
    created_at: datetime
    checksum: str  # stable dedupe key
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedDocument:
    """
    Canonical normalized representation used by the indexing pipeline.
    """

    ref: DocumentRef
    text: str
    pages: Optional[List[str]] = None  # for PDF page texts, if available


@dataclass(frozen=True)
class Chunk:
    """
    A semantically meaningful span of a normalized document.
    """

    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    # Optional anchors for citation
    page_number: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None


@dataclass(frozen=True)
class EmbeddedChunk:
    """
    A chunk paired with its embedding vector.
    """

    chunk: Chunk
    vector: List[float]


@dataclass(frozen=True)
class SearchQuery:
    """
    Semantic search query parameters.
    """

    query: str
    top_k: int = 10
    min_score: Optional[float] = None
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """
    A single search hit returned from the vector store.
    """

    chunk_id: str
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    vector: Optional[List[float]] = None




