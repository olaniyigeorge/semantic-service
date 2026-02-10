from __future__ import annotations

import hashlib
import re

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
