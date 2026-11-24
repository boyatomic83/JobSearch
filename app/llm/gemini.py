"""Google Gemini via the official google-genai SDK (the spec's original default).

Note: on 2026-11-24 the user's Google project returned 403 "free tier is not available for
this project", so the default provider was switched to Groq. This stays as a drop-in.
"""

from typing import Any

from google import genai
from google.genai import errors, types

from app.llm.base import LLMProvider, LLMResponse, RateLimited


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise ValueError("LLM_PROVIDER=gemini needs GEMINI_API_KEY")
        if not model:
            raise ValueError("LLM_MODEL is not set")
        self.model = model
        self._client = genai.Client(api_key=api_key)

    async def complete(
        self, *, system: str, messages: list[dict[str, Any]], json_mode: bool = False
    ) -> LLMResponse:
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            response_mime_type="application/json" if json_mode else None,
        )
        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except errors.APIError as exc:
            if exc.code == 429:
                raise RateLimited(str(exc)) from exc
            raise
        usage = resp.usage_metadata
        return LLMResponse(
            text=resp.text or "",
            prompt_tokens=(usage.prompt_token_count or 0) if usage else 0,
            completion_tokens=(usage.candidates_token_count or 0) if usage else 0,
        )
