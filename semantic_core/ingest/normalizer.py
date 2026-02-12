from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import uuid

from fastapi import HTTPException, UploadFile

from semantic_core.models import DocumentRef, NormalizedDocument
from semantic_core.ingest import readers

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

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





@dataclass(frozen=True)
class FileInput:
    """Framework-agnostic file container."""
    data: bytes
    filename: Optional[str] = None


class DocumentNormalizer:
    """
    Turns an IngestRequest (+ optional file) into a NormalizedDocument.
    """

    async def normalize_from_request(
        self,
        payload,
        *,
        upload_file: Optional[UploadFile] = None,
        file_input: Optional[FileInput] = None,
        created_at: Optional[datetime] = None,
    ):
        """
        FastAPI-friendly entrypoint. Accepts UploadFile or FileInput.
        Returns NormalizedDocument.
        """
        if upload_file and file_input:
            raise HTTPException(status_code=400, detail="Provide only one of upload_file or file_input")

        if upload_file is not None:
            data = await upload_file.read()
            file_input = FileInput(data=data, filename=upload_file.filename)

        return self.normalize(payload, file_input=file_input, created_at=created_at)

    def normalize(
        self,
        payload,
        *,
        file_input: Optional[FileInput] = None,
        created_at: Optional[datetime] = None,
    ):
        """
        Pure function-style normalization (sync) once file bytes are available.
        Returns NormalizedDocument.
        """
        # 1) choose text source (define a precedence rule)
        text: Optional[str] = None
        if payload.raw_text:
            text = payload.raw_text
        elif payload.json_text:
            text = payload.json_text
        elif file_input is not None:
            text = _extract_text_from_file(payload.doc_type, file_input.data)
        else:
            raise HTTPException(status_code=422, detail="Provide one of: raw_text, json_text, or file")

        # 2) normalize + validate
        text = normalize_text(text)
        if not text:
            raise HTTPException(status_code=400, detail="Extracted text is empty")

        # 3) build ids + checksum
        checksum = sha256(text)
        doc_id = str(uuid.uuid4())

        # 4) build ref + normalized doc
        doc_ref = DocumentRef(
            doc_id=doc_id,
            source=payload.source,
            doc_type=payload.doc_type,
            created_at=created_at or datetime.now(),
            checksum=checksum,
            extra={
                "filename": file_input.filename if file_input else None,
                "metadata": payload.metadata,
            },
        )

        return NormalizedDocument(ref=doc_ref, text=text)
