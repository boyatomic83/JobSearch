# bookworm (OpenSSL 3.0): the trixie-based slim image's larger TLS ClientHello timed out
# against PyPI from Codespaces during builds.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=60

WORKDIR /app

# Install deps first so code changes don't bust the dependency layer.
COPY pyproject.toml ./
RUN mkdir app && touch app/__init__.py && pip install . && rm -rf app

COPY . .
RUN pip install --no-deps . \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser /app
USER appuser

EXPOSE 8000
# Migrations run on every start; Alembic is a no-op when already at head.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
