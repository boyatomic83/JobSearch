# bookworm (OpenSSL 3.0): the trixie-based slim image's larger TLS ClientHello timed out
# against PyPI from Codespaces during builds.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=60 \
    EMBEDDING_CACHE_DIR=/app/.cache/fastembed \
    # glibc otherwise makes one malloc arena per thread (ONNX + asyncio.to_thread); measured
    # peak RSS while embedding 1,000 jobs: 476 MB → 440 MB. Matters on a 512 MB instance.
    MALLOC_ARENA_MAX=2

RUN useradd --create-home --uid 1000 appuser \
    && mkdir /app && chown appuser /app
WORKDIR /app

# Dependencies only (not our code), so code changes don't bust this layer.
COPY pyproject.toml ./
RUN mkdir app && touch app/__init__.py && pip install . && pip uninstall -y jobscout \
    && rm -rf app build *.egg-info

USER appuser
# Bake the embedding model into the image so cold starts don't download ~130 MB,
# then forbid Hub calls at runtime (the model is already on disk).
RUN python -c "from fastembed import TextEmbedding as T; T('BAAI/bge-small-en-v1.5', cache_dir='$EMBEDDING_CACHE_DIR')"
ENV HF_HUB_OFFLINE=1

COPY --chown=appuser . .

EXPOSE 8000
# Migrations run on every start; Alembic is a no-op when already at head.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
