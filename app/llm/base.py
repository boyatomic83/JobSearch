"""The provider interface: the only thing a new LLM vendor has to implement.

Providers do one raw call and translate vendor errors. Everything that should behave the
same for every vendor — caching, rate limiting, retries, JSON validation, logging — lives
in `app.llm.client.LLMClient`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A function call the model asked us to run (used by the agent in Phase 5)."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)


class RateLimited(Exception):
    """The provider said 429 / quota exceeded. `retry_after` is in seconds, if given."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def complete(
        self, *, system: str, messages: list[dict[str, Any]], json_mode: bool = False
    ) -> LLMResponse:
        """One chat completion.

        `messages` use a neutral format: [{"role": "user" | "assistant", "content": str}].
        `json_mode` asks the provider to return a single JSON object.
        """
