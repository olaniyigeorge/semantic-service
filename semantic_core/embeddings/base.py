from __future__ import annotations

from typing import List, Protocol, Sequence


class Embedder(Protocol):
    """
    Embedding backend interface used by the semantic pipelines.

    Implementations must return one vector per input text, in the same order.
    """

    @property
    def dim(self) -> int:
        """
        Dimensionality of the embedding vectors.
        """
        ...

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of input texts.
        """
        ...
