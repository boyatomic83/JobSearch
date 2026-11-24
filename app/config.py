"""Application settings, loaded from environment variables / `.env`.

All config goes through `get_settings()` so tests can override values in one place.
"""

from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The pgvector column size. Fixed so switching embedding providers never needs a migration:
# bge-small-en-v1.5 is natively 384-d, and gemini-embedding-001 can be asked for 384-d output.
EMBEDDING_DIM = 384

# libpq query params that asyncpg does not understand (Neon adds these to its URLs).
_LIBPQ_ONLY_PARAMS = {"sslmode", "channel_binding"}


def to_async_url(url: str) -> str:
    """Convert a libpq-style Postgres URL (as Neon/Render give you) into an asyncpg URL.

    Why: providers hand out `postgresql://...?sslmode=require&channel_binding=require`,
    but SQLAlchemy's asyncpg driver needs `postgresql+asyncpg://` and `ssl=require`.
    """
    if not url:
        return url
    parts = urlsplit(url)
    scheme = parts.scheme
    if scheme in ("postgres", "postgresql"):
        scheme = "postgresql+asyncpg"

    params = dict(parse_qsl(parts.query))
    sslmode = params.get("sslmode")
    params = {k: v for k, v in params.items() if k not in _LIBPQ_ONLY_PARAMS}
    if sslmode and sslmode != "disable" and "ssl" not in params:
        params["ssl"] = "require"

    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://jobscout:jobscout@localhost:5432/jobscout"
    gemini_api_key: str = ""
    groq_api_key: str = ""
    xai_api_key: str = ""
    llm_provider: str = "gemini"  # gemini | groq | xai
    llm_model: str = ""
    # Optional separate model for the chat agent's own turns. Groq rate-limits per model,
    # so this gives the agent its own tokens/minute budget (rerank etc. stay on LLM_MODEL).
    llm_agent_model: str = ""
    # Optional, for reasoning models (e.g. gpt-oss): "low" saves tokens on the free tier.
    llm_reasoning_effort: str = ""
    llm_rpm: int = 20  # our own cap, kept below the provider's free-tier requests/minute
    llm_tpm: int = 7000  # tokens/minute budget (Groq free tier: 8000); 0 disables
    # Estimated $ per 1M tokens, for cost reporting only (free tier is $0 in practice).
    llm_price_in_per_mtok: float = 0.0
    llm_price_out_per_mtok: float = 0.0
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Where fastembed keeps the ONNX model (baked into the Docker image at build time).
    embedding_cache_dir: str = ".cache/fastembed"
    telegram_bot_token: str = ""
    telegram_webhook_url: str = ""
    admin_api_key: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def async_database_url(self) -> str:
        return to_async_url(self.database_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
