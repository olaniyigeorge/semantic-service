from __future__ import annotations

from typing import Protocol, Dict, Any

from semantic_core.models import DocumentRef


class MetadataBuilder(Protocol):
    """
    Builds base metadata copied into every chunk for a given product.
    """

    def build_document_metadata(self, ref: DocumentRef) -> Dict[str, Any]:
        """Metadata copied into every chunk (filters rely on this)."""
        ...
