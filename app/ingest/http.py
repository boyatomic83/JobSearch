"""A small "polite" HTTP client shared by all connectors.

Why: we hit other people's free public APIs, so we (1) space out requests per host,
(2) retry transient failures (429 / 5xx / timeouts) with exponential backoff and honour
Retry-After, and (3) fail fast on permanent errors like 404 (wrong company slug).
"""

import asyncio
import logging
import random
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "JobScout/0.1 (portfolio job-matching project)"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class FetchError(Exception):
    """Raised when a URL could not be fetched after retries (or failed permanently)."""


class PoliteClient:
    def __init__(
        self,
        *,
        min_interval: float = 1.0,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            follow_redirects=True,
            transport=transport,
        )
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._sleep = sleep
        self._host_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._last_request: dict[str, float] = {}

    async def __aenter__(self) -> "PoliteClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self._client.aclose()

    async def _wait_for_slot(self, host: str) -> None:
        """At most one request per `min_interval` seconds per host."""
        async with self._host_locks[host]:
            elapsed = time.monotonic() - self._last_request.get(host, 0.0)
            if elapsed < self._min_interval:
                await self._sleep(self._min_interval - elapsed)
            self._last_request[host] = time.monotonic()

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after and retry_after.isdigit():
            return min(float(retry_after), 60.0)
        return self._backoff_base * (2**attempt) + random.uniform(0, self._backoff_base)

    async def get_json(self, url: str, params: dict[str, str] | None = None) -> Any:
        host = urlsplit(url).netloc
        for attempt in range(self._max_retries + 1):
            await self._wait_for_slot(host)
            retry_after = None
            try:
                resp = await self._client.get(url, params=params)
            except httpx.TransportError as exc:  # timeouts, connection resets, DNS…
                reason = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code < 400:
                    return resp.json()
                if resp.status_code not in RETRYABLE_STATUS:
                    raise FetchError(f"GET {url} -> HTTP {resp.status_code} (not retrying)")
                reason = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After")

            if attempt == self._max_retries:
                raise FetchError(f"GET {url} failed after {attempt + 1} attempts: {reason}")
            delay = self._backoff(attempt, retry_after)
            logger.warning("GET %s: %s, retrying in %.1fs", url, reason, delay)
            await self._sleep(delay)
        raise AssertionError("unreachable")
