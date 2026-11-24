"""Google Gemini via the official google-genai SDK (the spec's original default).

Note: on 2026-11-24 the user's Google project returned 403 "free tier is not available for
this project", so the default provider was switched to Groq. This stays as a drop-in; its
tool-calling mapping is unit-tested but has not been exercised against the live API.
"""

import json
import uuid
from typing import Any

from google import genai
from google.genai import errors, types

from app.llm.base import LLMProvider, LLMResponse, RateLimited, ToolCall, ToolSpec


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
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        json_mode: bool = False,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            response_mime_type="application/json" if json_mode else None,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name,
                            description=t.description,
                            parameters_json_schema=t.parameters,
                        )
                        for t in tools
                    ]
                )
            ]
            if tools
            else None,
        )
        contents = [to_gemini_content(m) for m in messages]
        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        except errors.APIError as exc:
            if exc.code == 429:
                raise RateLimited(str(exc)) from exc
            raise
        usage = resp.usage_metadata
        calls = [
            ToolCall(
                id=fc.id or uuid.uuid4().hex, name=fc.name or "", arguments=dict(fc.args or {})
            )
            for fc in resp.function_calls or []
        ]
        text_parts = [
            p.text
            for c in resp.candidates or []
            for p in (c.content.parts if c.content and c.content.parts else [])
            if p.text
        ]
        return LLMResponse(
            text="".join(text_parts),
            prompt_tokens=(usage.prompt_token_count or 0) if usage else 0,
            completion_tokens=(usage.candidates_token_count or 0) if usage else 0,
            tool_calls=calls,
        )


def to_gemini_content(m: dict[str, Any]) -> types.Content:
    """Neutral message → Gemini Content (tool results go back as function_response parts)."""
    if m["role"] == "tool":
        try:
            result = json.loads(m["content"])
        except json.JSONDecodeError:
            result = m["content"]
        return types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=m["tool_call_id"], name=m["name"], response={"result": result}
                    )
                )
            ],
        )
    parts = [types.Part(text=m["content"])] if m.get("content") else []
    for c in m.get("tool_calls") or []:
        parts.append(
            types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.arguments))
        )
    return types.Content(role="model" if m["role"] == "assistant" else "user", parts=parts)
