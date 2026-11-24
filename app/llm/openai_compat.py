"""Any OpenAI-compatible chat API. Groq (GroqCloud) and xAI (Grok) both speak it, so one
small class covers both — only the base URL and key differ.
"""

from typing import Any

import openai

from app.llm.base import LLMProvider, LLMResponse, RateLimited

BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "xai": "https://api.x.ai/v1",
}


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        reasoning_effort: str = "",
    ) -> None:
        if not api_key:
            raise ValueError(f"LLM_PROVIDER={name} needs {name.upper()}_API_KEY")
        if not model:
            raise ValueError("LLM_MODEL is not set")
        self.name = name
        self.model = model
        # Reasoning models think before answering; those tokens count against the free
        # tier's tokens-per-minute budget, so we let config turn the effort down.
        self._reasoning_effort = reasoning_effort
        # max_retries=0: retries/backoff are handled once, uniformly, in LLMClient.
        self._client = openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url or BASE_URLS[name], max_retries=0, timeout=60
        )

    async def complete(
        self, *, system: str, messages: list[dict[str, Any]], json_mode: bool = False
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
                temperature=0.2,  # low: extraction/ranking should be repeatable
                **kwargs,
            )
        except openai.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after") if exc.response else None
            raise RateLimited(str(exc), float(retry_after) if retry_after else None) from exc
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
