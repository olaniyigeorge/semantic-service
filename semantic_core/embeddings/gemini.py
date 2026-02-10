from __future__ import annotations

import os
from typing import List, Sequence

from google.genai import types
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


from .base import Embedder


class GeminiEmbeddings(Embedder):
    """
    Placeholder Gemini embeddings backend.
    """

    def __init__(self, model_name: str = "gemini-embedding-001") -> None:
        self.model_name = model_name

    @property
    def dim(self) -> int:
        # TODO: return the actual embedding dimensionality once known
        return 0

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Synchronous wrapper around the Gemini embeddings API.
        """
        from google import genai
        API_KEY = os.getenv("GEMINI_API_KEY")
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        client = genai.Client(api_key=API_KEY)
        contents = list(texts)
        if not contents:
            contents = ["placeholder text for empty batch"]

        result = client.models.embed_content(
            model=self.model_name,
            contents=contents,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )

        df = pd.DataFrame(
            cosine_similarity([e.values for e in result.embeddings]),
            index=texts,
            columns=texts,
        )

        print()
        print(df)
        print()

        embeddings = result.embeddings
        return embeddings

        


