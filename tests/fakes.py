"""Test doubles (synthetic, deterministic) shared across tests."""

import hashlib
import math
import re

from app.matching.embeddings import Embedder


class FakeEmbedder(Embedder):
    """Bag-of-words hashed into 384 buckets: texts sharing words get similar vectors.

    Deterministic and instant, so DB tests can exercise vector search without the model.
    """

    model_name = "fake-bow"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
