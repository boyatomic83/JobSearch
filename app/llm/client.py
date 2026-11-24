"""LLMClient: the one place every LLM request goes through.

Why a wrapper instead of calling providers directly:
- **Free-tier safety:** a process-wide rate limiter (requests/minute) and exponential
  backoff on 429s, so bursts never burn the daily quota on retries.
- **Cache:** identical (model, prompt, schema) requests are served from `llm_cache`.
- **Validation:** structured output is parsed with Pydantic; on invalid JSON we send the
  validation error back once and ask the model to fix it.
- **Observability:** every attempt (including cache hits and failures) is a row in
  `llm_calls` with tokens, latency and estimated cost.
"""

import asyncio
import hashlib
import json
import logging
import time
from decimal import Decimal
from functools import lru_cache
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.db.models import LLMCache, LLMCall
from app.llm.base import LLMProvider, LLMResponse, RateLimited

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The LLM could not produce a valid answer (after retries / repair)."""


class RateLimiter:
    """At most `rpm` requests per minute from this process, spaced evenly."""

    def __init__(self, rpm: int, sleep=asyncio.sleep) -> None:
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_at = 0.0
        self._sleep = sleep

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._next_at > now:
                await self._sleep(self._next_at - now)
            self._next_at = max(now, self._next_at) + self._interval


class LLMClient:
    def __init__(
        self,
        provider: LLMProvider,
        sessionmaker: async_sessionmaker[AsyncSession],
        *,
        rpm: int = 20,
        max_retries: int = 4,
        price_in_per_mtok: float = 0.0,
        price_out_per_mtok: float = 0.0,
        sleep=asyncio.sleep,
    ) -> None:
        self.provider = provider
        self._sessionmaker = sessionmaker
        self._limiter = RateLimiter(rpm, sleep)
        self._max_retries = max_retries
        self._price_in = price_in_per_mtok
        self._price_out = price_out_per_mtok
        self._sleep = sleep

    # ---- public API ---------------------------------------------------------------

    async def structured(
        self,
        *,
        purpose: str,
        system: str,
        prompt: str,
        schema: type[T],
        user_id: int | None = None,
        use_cache: bool = True,
    ) -> T:
        """Ask for JSON matching `schema` and return it validated."""
        schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        system = (
            f"{system}\n\nRespond with a single JSON object that matches this JSON Schema "
            f"exactly (no prose, no markdown):\n{schema_json}"
        )
        key = self._cache_key(system, prompt)
        if use_cache and (hit := await self._cache_get(key)) is not None:
            try:
                result = schema.model_validate(hit)
                await self._log(purpose, user_id, LLMResponse(text=""), 0, cached=True)
                return result
            except ValidationError:
                logger.warning("stale cache entry for %s, ignoring", purpose)

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        for attempt in range(2):  # first try + one repair
            resp = await self._call(purpose, user_id, system, messages, json_mode=True)
            try:
                result = schema.model_validate_json(_strip_fences(resp.text))
            except ValidationError as exc:
                if attempt == 1:
                    raise LLMError(f"{purpose}: invalid JSON after repair: {exc}") from exc
                logger.warning("%s: invalid JSON, asking model to repair", purpose)
                messages += [
                    {"role": "assistant", "content": resp.text},
                    {
                        "role": "user",
                        "content": f"That JSON was invalid:\n{exc}\nReturn the corrected "
                        "JSON object only.",
                    },
                ]
                continue
            if use_cache:
                await self._cache_put(key, result.model_dump(mode="json"))
            return result
        raise AssertionError("unreachable")

    # ---- internals ----------------------------------------------------------------

    async def _call(
        self,
        purpose: str,
        user_id: int | None,
        system: str,
        messages: list[dict[str, Any]],
        json_mode: bool,
    ) -> LLMResponse:
        for attempt in range(self._max_retries + 1):
            await self._limiter.wait()
            started = time.monotonic()
            try:
                resp = await self.provider.complete(
                    system=system, messages=messages, json_mode=json_mode
                )
            except RateLimited as exc:
                latency = int((time.monotonic() - started) * 1000)
                await self._log(purpose, user_id, None, latency, status="rate_limited")
                if attempt == self._max_retries:
                    raise LLMError(f"{purpose}: still rate-limited after retries") from exc
                delay = exc.retry_after or 2 ** (attempt + 1)
                logger.warning("%s: rate-limited, retrying in %.0fs", purpose, delay)
                await self._sleep(min(delay, 60))
                continue
            except Exception as exc:
                latency = int((time.monotonic() - started) * 1000)
                await self._log(purpose, user_id, None, latency, status="error", error=exc)
                raise LLMError(f"{purpose}: {type(exc).__name__}: {exc}") from exc
            latency = int((time.monotonic() - started) * 1000)
            await self._log(purpose, user_id, resp, latency)
            return resp
        raise AssertionError("unreachable")

    def _cache_key(self, system: str, prompt: str) -> str:
        raw = json.dumps([self.provider.name, self.provider.model, system, prompt])
        return hashlib.sha256(raw.encode()).hexdigest()

    async def _cache_get(self, key: str) -> dict | None:
        async with self._sessionmaker() as session:
            return await session.scalar(select(LLMCache.response).where(LLMCache.key == key))

    async def _cache_put(self, key: str, response: dict) -> None:
        async with self._sessionmaker() as session, session.begin():
            await session.execute(
                insert(LLMCache)
                .values(key=key, model=self.provider.model, response=response)
                .on_conflict_do_nothing()
            )

    def _cost(self, resp: LLMResponse) -> Decimal:
        cost = resp.prompt_tokens * self._price_in + resp.completion_tokens * self._price_out
        return Decimal(str(round(cost / 1_000_000, 6)))

    async def _log(
        self,
        purpose: str,
        user_id: int | None,
        resp: LLMResponse | None,
        latency_ms: int,
        *,
        cached: bool = False,
        status: str = "ok",
        error: Exception | None = None,
    ) -> None:
        """Own short transaction, so calls are logged even when the caller's work fails."""
        resp = resp or LLMResponse(text="")
        try:
            async with self._sessionmaker() as session, session.begin():
                session.add(
                    LLMCall(
                        user_id=user_id,
                        provider=self.provider.name,
                        model=self.provider.model,
                        purpose=purpose,
                        prompt_tokens=resp.prompt_tokens,
                        completion_tokens=resp.completion_tokens,
                        cost_usd=self._cost(resp),
                        latency_ms=latency_ms,
                        cached=cached,
                        status=status,
                        error=f"{type(error).__name__}: {error}"[:1000] if error else None,
                    )
                )
        except Exception:  # logging must never break the request
            logger.exception("failed to log llm call")


def _strip_fences(text: str) -> str:
    """Some models wrap JSON in ```json fences even in JSON mode."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def build_provider() -> LLMProvider:
    s = get_settings()
    if s.llm_provider == "gemini":
        from app.llm.gemini import GeminiProvider

        return GeminiProvider(s.llm_model, s.gemini_api_key)
    if s.llm_provider in ("groq", "xai"):
        from app.llm.openai_compat import OpenAICompatibleProvider

        key = s.groq_api_key if s.llm_provider == "groq" else s.xai_api_key
        return OpenAICompatibleProvider(
            s.llm_provider, s.llm_model, key, reasoning_effort=s.llm_reasoning_effort
        )
    raise ValueError(f"unknown LLM_PROVIDER {s.llm_provider!r}")


@lru_cache
def get_llm_client() -> LLMClient:
    """Process-wide client (one rate limiter shared by every request)."""
    from app.db.session import get_sessionmaker

    s = get_settings()
    return LLMClient(
        build_provider(),
        get_sessionmaker(),
        rpm=s.llm_rpm,
        price_in_per_mtok=s.llm_price_in_per_mtok,
        price_out_per_mtok=s.llm_price_out_per_mtok,
    )
