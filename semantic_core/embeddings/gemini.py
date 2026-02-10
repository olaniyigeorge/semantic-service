from __future__ import annotations

from typing import List, Sequence

from .base import Embedder


class GeminiEmbeddings(Embedder):
    """
    Placeholder Gemini embeddings backend.

    Wire this up to Google's Gemini API in a future iteration.
    """

    def __init__(self, model_name: str = "text-embedding-004") -> None:
        self.model_name = model_name

    @property
    def dim(self) -> int:
        # TODO: return the actual embedding dimensionality once known
        return 0

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Synchronous wrapper around the Gemini embeddings API.

        NOTE: This is a placeholder implementation; wire up proper auth and
        error handling before using in production.
        """
        from google import genai

        client = genai.Client()
        contents = list(texts)
        if not contents:
            contents = ["placeholder text for empty batch"]

        result = client.models.embed_content(
            model=self.model_name,
            contents=contents,
        )

        # The exact shape of `result.embeddings` depends on the SDK version.
        # For now we assume it is already a List[List[float]].
        embeddings = result.embeddings
        return embeddings  # type: ignore[return-value]

        


