from __future__ import annotations

from typing import Dict, Any

from semantic_core.models import DocumentRef
from semantic_core.metadata.base import MetadataBuilder


class CoopWiseMetadataBuilder(MetadataBuilder):
    def __init__(self, cooperative_id: str, group_id: str | None = None) -> None:
        self.cooperative_id = cooperative_id
        self.group_id = group_id

    def build_document_metadata(self, ref: DocumentRef) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "product": "coopwise",
            "doc_type": ref.doc_type,
            "source": ref.source,
            "checksum": ref.checksum,
            "cooperative_id": self.cooperative_id,
        }
        if self.group_id:
            meta["group_id"] = self.group_id
        meta.update(ref.extra or {})
        return meta

from __future__ import annotations

from typing import Any, Dict


def normalize_coopwise_metadata(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform Coopwise-specific metadata into a normalized schema.
    """
    # TODO: implement Coopwise-specific normalization rules
    return raw


