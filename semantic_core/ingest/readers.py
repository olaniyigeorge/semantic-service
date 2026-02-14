from __future__ import annotations

import json
from io import BytesIO
from typing import Any


def read_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def read_json(data: bytes) -> str:
    obj: Any = json.loads(data.decode("utf-8", errors="ignore"))
    # common shapes: {"text": "..."} or {"data": {"text": "..."}}
    if isinstance(obj, dict):
        if "text" in obj and isinstance(obj["text"], str):
            return obj["text"]
        if (
            "data" in obj
            and isinstance(obj["data"], dict)
            and isinstance(obj["data"].get("text"), str)
        ):
            return obj["data"]["text"]
    # fallback: stringify
    return json.dumps(obj, ensure_ascii=False)


def read_csv(data: bytes) -> str:
    # lightweight: convert rows to a text block (no pandas needed)
    text = data.decode("utf-8", errors="ignore")
    return text


def read_docx(data: bytes) -> str:
    from docx import Document  # python-docx

    doc = Document(BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def read_pdf(data: bytes) -> str:
    # pypdf is a common choice
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)
