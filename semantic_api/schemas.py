from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    """
    API-level representation of a document submitted for indexing.
    """

    id: str = Field(..., description="Unique identifier of the document")
    content: str = Field(..., description="Raw text content of the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for filtering / ranking",
    )





DocType = Literal["pdf", "txt", "docx", "csv", "json", "raw", "html"]
Product = Literal["coopwise", "winnov8"]


class IngestRequest(BaseModel):
    source: str = Field(..., examples=["upload", "admin", "api"])
    doc_type: DocType
    product: Product
    raw_text: Optional[str] = None
    json_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    checksum: str
