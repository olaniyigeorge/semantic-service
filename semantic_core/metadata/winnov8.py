from __future__ import annotations

from typing import Dict, Any

from semantic_core.models import DocumentRef
from semantic_core.metadata.base import MetadataBuilder


class Winnov8MetadataBuilder(MetadataBuilder):
    def __init__(self, owner_id: str, entity_type: str) -> None:
        self.owner_id = owner_id
        self.entity_type = entity_type  # resume/project_doc/request/etc.

    def build_document_metadata(self, ref: DocumentRef) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "product": "winnov8",
            "doc_type": ref.doc_type,
            "source": ref.source,
            "checksum": ref.checksum,
            "owner_id": self.owner_id,
            "entity_type": self.entity_type,
        }
        meta.update(ref.extra or {})
        return meta

from __future__ import annotations

from typing import Any, Dict


def normalize_winnov8_metadata(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform Winnov8-specific metadata into a normalized schema.
    """
    # TODO: implement Winnov8-specific normalization rules
    return raw


