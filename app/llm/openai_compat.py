"""Any OpenAI-compatible chat API. Groq (GroqCloud) and xAI (Grok) both speak it, so one
small class covers both — only the base URL and key differ.
"""

import json
from typing import Any

import openai

from app.llm.base import LLMProvider, LLMResponse, RateLimited, ToolCall, ToolSpec

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
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        json_mode: bool = False,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name, "description": t.description, "parameters": t.parameters
                    },
                }
                for t in tools
            ]  # fmt: skip
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, *map(_to_openai, messages)],
                temperature=0.2,  # low: extraction/ranking should be repeatable
                **kwargs,
            )
        except openai.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after") if exc.response else None
            raise RateLimited(str(exc), float(retry_after) if retry_after else None) from exc
        usage = resp.usage
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=_parse_args(c.function.arguments))
            for c in msg.tool_calls or []
        ]
        return LLMResponse(
            text=msg.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            tool_calls=calls,
        )


def _parse_args(raw: str | None) -> dict[str, Any]:
    """Models occasionally emit invalid JSON arguments; surface that as a tool error later."""
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"__invalid_json__": raw}
    return args if isinstance(args, dict) else {"__invalid_json__": raw}


def _to_openai(m: dict[str, Any]) -> dict[str, Any]:
    """Neutral message → OpenAI chat format."""
    if m["role"] == "tool":
        return {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
    if m["role"] == "assistant" and m.get("tool_calls"):
        return {
            "role": "assistant",
            "content": m.get("content") or None,
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in m["tool_calls"]
            ],
        }
    return {"role": m["role"], "content": m["content"]}
