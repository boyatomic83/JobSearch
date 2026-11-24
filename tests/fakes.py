"""Test doubles (synthetic, deterministic) shared across tests."""

import hashlib
import math
import re

from app.matching.embeddings import Embedder


class FakeEmbedder(Embedder):
    """Bag-of-words hashed into 384 buckets: texts sharing words get similar vectors.

    Deterministic and instant, so DB tests can exercise vector search without the model.
    """

    model_name = "fake-bow"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeLLMProvider:
    """Scripted LLM: returns queued replies in order (str, Exception, or callable(messages)).

    Records every call so tests can assert on prompts. Synthetic — no network.
    """

    name = "fake"
    model = "fake-model"

    def __init__(self, replies: list | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[dict] = []

    async def complete(self, *, system, messages, json_mode=False, tools=None):
        """A reply may be: str (text), LLMResponse (e.g. with tool_calls), an Exception to
        raise, or a callable(messages) returning either."""
        from app.llm.base import LLMResponse

        self.calls.append(
            {"system": system, "messages": list(messages), "json_mode": json_mode, "tools": tools}
        )
        if not self.replies:
            raise AssertionError("FakeLLMProvider: no scripted reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            reply = reply(messages)
        if isinstance(reply, LLMResponse):
            return reply
        return LLMResponse(text=reply, prompt_tokens=100, completion_tokens=20)


def tool_call(name: str, call_id: str = "c1", **arguments):
    """A scripted LLM reply that calls one tool."""
    from app.llm.base import LLMResponse, ToolCall

    call = ToolCall(id=call_id, name=name, arguments=arguments)
    return LLMResponse(text="", prompt_tokens=100, completion_tokens=20, tool_calls=[call])


async def no_sleep(_seconds: float) -> None:
    pass


def make_llm(sessionmaker, replies=None, **kw):
    """An LLMClient around a FakeLLMProvider with no real waiting."""
    from app.llm.client import LLMClient

    provider = FakeLLMProvider(replies)
    client = LLMClient(provider, sessionmaker, rpm=0, sleep=no_sleep, **kw)
    return client, provider


def make_pdf(text: str) -> bytes:
    """Render plain text into a real (text-layer) PDF, for resume upload tests."""
    import io

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for line in text.splitlines():
        c.drawString(40, y, line)
        y -= 14
        if y < 40:
            c.showPage()
            y = 800
    c.save()
    return buf.getvalue()


# Synthetic resume used across tests (not a real person).
SAMPLE_RESUME = """Priya Sharma (synthetic test resume)
Bangalore, India | priya@example.com
Senior Backend Engineer, Acme Payments (2021-present)
- Built Python FastAPI services handling 2k requests/second on PostgreSQL
- Led migration of batch jobs to Kubernetes; cut infra cost by 30%
Software Engineer, Globex (2018-2021)
- Developed Django REST APIs and Celery pipelines
Skills: Python, FastAPI, Django, PostgreSQL, Redis, Kubernetes, AWS, Docker
Education: B.Tech Computer Science, IIT Delhi, 2018
"""

SAMPLE_PROFILE = {
    "name": "Priya Sharma",
    "location": "Bangalore, India",
    "current_title": "Senior Backend Engineer",
    "years_experience": 7,
    "seniority": "senior",
    "roles": ["Senior Backend Engineer", "Software Engineer"],
    "skills": ["Python", "FastAPI", "Django", "PostgreSQL", "Redis", "Kubernetes", "AWS"],
    "education": [{"degree": "B.Tech", "field": "Computer Science",
                   "institution": "IIT Delhi", "year": 2018}],
    "summary": "Backend engineer with 7 years building Python APIs on PostgreSQL.",
}  # fmt: skip
