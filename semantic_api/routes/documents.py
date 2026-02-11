# semantic_api/routes/documents.py
from __future__ import annotations
from datetime import datetime 
import logging
import uuid
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi import Body, Request

from semantic_api.deps import get_index_pipeline

from semantic_core.ingest.normalize import normalize_text, sha256
from semantic_api.schemas import DocumentIn, IngestRequest, IngestResponse
from semantic_core.ingest import readers
from semantic_core.pipeline.indexer import IndexPipeline
from semantic_core.models import NormalizedDocument, DocumentRef

router = APIRouter(prefix="/v1/documents", tags=["documents"])


def _extract_text_from_file(doc_type: str, data: bytes) -> str:
    match doc_type:
        case "txt":
            return readers.read_txt(data)
        case "json":
            return readers.read_json(data)
        case "csv":
            return readers.read_csv(data)
        case "docx":
            return readers.read_docx(data)
        case "pdf":
            return readers.read_pdf(data)
        case "html":
            return readers.read_txt(data)
        case _:
            raise HTTPException(status_code=400, detail=f"Unsupported doc_type: {doc_type}")


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    # --- JSON body path (application/json) ---
    # request: Request,
    payload: Optional[IngestRequest] = Body(default=None),
    # --- multipart/form-data path ---
    source: Optional[str] = Form(default=None),
    doc_type: Optional[str] = Form(default=None),
    product: Optional[str] = Form(default=None),
    raw_text: Optional[str] = Form(default=None),
    json_text: Optional[str] = Form(default=None),
    metadata: Optional[str] = Form(default=None), 
    file: Optional[UploadFile] = File(default=None),
    pipeline: IndexPipeline = Depends(get_index_pipeline),
) -> IngestResponse:
    """
    Ingest a document from raw text, JSON text, or uploaded file.
    Normalizes text, computes checksum, runs IndexPipeline, returns doc_id + checksum.
    """
    # print("\n\n\n\nContent-Type: %s", request.headers.get("content-type"))
    
    # Decide which input mode is used
    if payload is None:
        # Multipart mode
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
            doc_type=doc_type,  # validated by model literals if you import same types
            product=product,
            raw_text=raw_text,
            json_text=json_text,
            metadata=meta_dict,
        )
        logging.info("\nReceived multipart ingest request: %s", payload, "\n")
    
    print(f"\nReceived payload: {payload}\n")
    # Build text content
    text: Optional[str] = None

    if payload.raw_text:
        text = payload.raw_text
    elif payload.json_text:
        text = payload.json_text
    elif file is not None:
        data = await file.read()
        text = _extract_text_from_file(payload.doc_type, data)
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide one of: raw_text, json_text, or file",
        )

    text = normalize_text(text)
    if not text:
        raise HTTPException(status_code=400, detail="Extracted text is empty")

    checksum = sha256(text)
    doc_id = str(uuid.uuid4())

    # ---- Create a normalized doc for your pipeline ----
    # Adjust these to your actual semantic_core models.
    # The important part is: doc.ref is used by metadata builder.
    doc_ref = DocumentRef(
        doc_id=doc_id,
        source=payload.source,
        doc_type=payload.doc_type,
        created_at=datetime.now(),
        checksum=checksum,
        extra ={"filename": file.filename if file else None, "metadata": payload.metadata},
    )

    normalized = NormalizedDocument(
        ref=doc_ref,
        text=text,
    )

    # # ---- Index: chunk -> embed -> upsert ----
    try:
        n_chunks = pipeline.index(normalized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}") from e

    # Optionally: persist doc_ref in DB here (recommended)
    # e.g. documents_repo.create(doc_ref, n_chunks=n_chunks)

    # print(f"Document Ref {doc_ref} \n\n\n\nwith normalised {normalized}")

    return IngestResponse(doc_id=doc_id, checksum=checksum)


@router.post(
    "/index_documents",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit documents for indexing",
)
async def index_documents(documents: List[DocumentIn]) -> dict:
    """
    Enqueue documents for ingestion and indexing.

    Actual indexing is not implemented yet.
    """
    # TODO: wire up IndexPipeline + concrete ingestor/chunker/metadata
    return {"received": len(documents)}


