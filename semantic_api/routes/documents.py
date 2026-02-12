from __future__ import annotations
from datetime import datetime 
import logging
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile, status

from semantic_api.deps import get_document_normalizer, get_index_pipeline 
from semantic_api.schemas import IngestRequest, IngestResponse
from semantic_core.ingest.normalizer import DocumentNormalizer
from semantic_core.pipeline.indexer import IndexPipeline


router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    payload: Optional[IngestRequest] = Body(default=None),
    source: Optional[str] = Form(default=None),
    doc_type: Optional[str] = Form(default=None),
    product: Optional[str] = Form(default=None),
    raw_text: Optional[str] = Form(default=None),
    json_text: Optional[str] = Form(default=None),
    metadata: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    pipeline: IndexPipeline = Depends(get_index_pipeline),
    normalizer: DocumentNormalizer = Depends(get_document_normalizer),
) -> IngestResponse:

    if payload is None:
        # Multipart mode validation
        if not (source and doc_type and product):
            raise HTTPException(status_code=422, detail="Missing required fields: source, doc_type, product")

        meta_dict: dict[str, Any] = {}
        if metadata:
            import json as _json
            try:
                meta_dict = _json.loads(metadata)
                if not isinstance(meta_dict, dict):
                    raise ValueError("metadata must be a JSON object")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {e}") from e

        payload = IngestRequest(
            source=source,
            doc_type=doc_type,
            product=product,
            raw_text=raw_text,
            json_text=json_text,
            metadata=meta_dict,
        )

        logging.info("Received multipart ingest request: %s", payload)

    normalized = await normalizer.normalize_from_request(payload, upload_file=file)

    try:
        n_chunks = pipeline.index(normalized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}") from e

    return IngestResponse(doc_id=normalized.ref.doc_id, checksum=normalized.ref.checksum)
