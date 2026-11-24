"""Embeddings behind a tiny interface so the provider is a config switch.

- `local` (default): BAAI/bge-small-en-v1.5 on CPU via fastembed (ONNX Runtime). Free, no
  rate limits, 384-d. Loaded once per process, lazily.
  Why fastembed instead of sentence-transformers: same model and identical vectors (we
  measured cosine 1.0000), but no PyTorch. Importing torch alone used 440 MB, so
  sentence-transformers could not fit Render's 512 MB free tier at any batch size.
- `gemini`: gemini-embedding-001 asked for 384-d output so the pgvector column is the same.

All vectors are L2-normalised, so cosine similarity == dot product and pgvector's `<=>`
(cosine distance) behaves the same for both providers.
"""

import logging
import math
import time
from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import EMBEDDING_DIM, get_settings

logger = logging.getLogger(__name__)


class Embedder(ABC):
    model_name: str
    dim: int = EMBEDDING_DIM

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed things being searched over (job postings)."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed the thing searching (a resume profile or a chat query)."""


class LocalEmbedder(Embedder):
    # bge models are trained with this instruction on the *query* side of retrieval;
    # documents are embedded as-is.
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    # Peak RAM grows with batch size (attention over 512 tokens per text). Measured peaks
    # for the whole process: batch 4 ≈ 410 MB, batch 8 ≈ 500 MB, batch 32 ≈ 850 MB.
    BATCH_SIZE = 4

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding  # imported here so tests can run without it

        self.model_name = model_name
        started = time.monotonic()
        self._model = TextEmbedding(model_name, cache_dir=cache_dir)
        logger.info("loaded %s in %.1fs", model_name, time.monotonic() - started)
        actual = len(self.embed_documents(["dimension check"])[0])
        if actual != self.dim:
            raise ValueError(f"{model_name} produces {actual}-d vectors, DB expects {self.dim}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.embed(texts, batch_size=self.BATCH_SIZE)
        return [_normalise(v.tolist()) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([self.QUERY_INSTRUCTION + text])[0]


class GeminiEmbedder(Embedder):
    BATCH = 100  # max texts per embed_content request

    def __init__(self, model_name: str, api_key: str, max_retries: int = 5) -> None:
        from google import genai

        if not api_key:
            raise ValueError("EMBEDDING_PROVIDER=gemini needs GEMINI_API_KEY")
        self.model_name = model_name
        self._client = genai.Client(api_key=api_key)
        self._max_retries = max_retries

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import errors, types

        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        out: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH):
            batch = texts[start : start + self.BATCH]
            for attempt in range(self._max_retries + 1):
                try:
                    resp = self._client.models.embed_content(
                        model=self.model_name, contents=batch, config=config
                    )
                    break
                except errors.APIError as exc:  # 429 = free-tier quota; back off and retry
                    if exc.code != 429 or attempt == self._max_retries:
                        raise
                    delay = 2**attempt * 5
                    logger.warning("gemini embeddings rate-limited, sleeping %ss", delay)
                    time.sleep(delay)
            # Truncated (non-3072-d) Gemini vectors are not normalised, so we do it.
            out += [_normalise(e.values) for e in resp.embeddings]
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@lru_cache
def get_embedder() -> Embedder:
    """The process-wide embedder (created on first use, then reused)."""
    s = get_settings()
    if s.embedding_provider == "local":
        return LocalEmbedder(s.embedding_model, cache_dir=s.embedding_cache_dir)
    if s.embedding_provider == "gemini":
        model = s.embedding_model if "gemini" in s.embedding_model else "gemini-embedding-001"
        return GeminiEmbedder(model, s.gemini_api_key)
    raise ValueError(f"unknown EMBEDDING_PROVIDER {s.embedding_provider!r}")
