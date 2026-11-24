"""LLMClient behaviour with a scripted fake provider (no network)."""

import json

import pytest
from pydantic import BaseModel
from sqlalchemy import select

from app.db.models import LLMCall
from app.llm.base import RateLimited
from app.llm.client import LLMError, RateLimiter, _strip_fences
from tests.fakes import make_llm


class Answer(BaseModel):
    value: int


async def _calls(sessionmaker) -> list[LLMCall]:
    async with sessionmaker() as s:
        return list((await s.scalars(select(LLMCall).order_by(LLMCall.id))).all())


async def test_structured_validates_and_logs(sessionmaker):
    llm, provider = make_llm(sessionmaker, ['{"value": 42}'])
    result = await llm.structured(purpose="t", system="sys", prompt="q", schema=Answer)
    assert result == Answer(value=42)
    assert provider.calls[0]["json_mode"] is True
    assert '"value"' in provider.calls[0]["system"]  # JSON schema is in the prompt
    [call] = await _calls(sessionmaker)
    assert (call.purpose, call.status, call.prompt_tokens, call.cached) == ("t", "ok", 100, False)


async def test_cache_hit_skips_provider_but_is_logged(sessionmaker):
    llm, provider = make_llm(sessionmaker, ['{"value": 1}'])
    await llm.structured(purpose="t", system="s", prompt="same", schema=Answer)
    again = await llm.structured(purpose="t", system="s", prompt="same", schema=Answer)
    assert again.value == 1
    assert len(provider.calls) == 1
    assert [c.cached for c in await _calls(sessionmaker)] == [False, True]


async def test_invalid_json_gets_one_repair_round(sessionmaker):
    llm, provider = make_llm(sessionmaker, ['{"value": "not a number"}', '{"value": 7}'])
    assert (await llm.structured(purpose="t", system="s", prompt="q", schema=Answer)).value == 7
    repair_msg = provider.calls[1]["messages"][-1]["content"]
    assert "invalid" in repair_msg


async def test_gives_up_after_failed_repair(sessionmaker):
    llm, _ = make_llm(sessionmaker, ["nope", "still nope"])
    with pytest.raises(LLMError, match="invalid JSON"):
        await llm.structured(purpose="t", system="s", prompt="q", schema=Answer)


async def test_retries_rate_limits_with_backoff(sessionmaker):
    sleeps: list[float] = []

    async def record(seconds):
        sleeps.append(seconds)

    llm, _ = make_llm(
        sessionmaker,
        [RateLimited("429"), RateLimited("429", retry_after=9), '{"value": 3}'],
    )
    llm._sleep = record
    assert (await llm.structured(purpose="t", system="s", prompt="q", schema=Answer)).value == 3
    assert sleeps == [2, 9]  # exponential first, then the server's Retry-After
    statuses = [c.status for c in await _calls(sessionmaker)]
    assert statuses == ["rate_limited", "rate_limited", "ok"]


async def test_provider_error_is_logged_and_wrapped(sessionmaker):
    llm, _ = make_llm(sessionmaker, [RuntimeError("boom")])
    with pytest.raises(LLMError, match="boom"):
        await llm.structured(purpose="t", system="s", prompt="q", schema=Answer)
    [call] = await _calls(sessionmaker)
    assert call.status == "error" and "boom" in call.error


async def test_cost_estimate_uses_configured_prices(sessionmaker):
    llm, _ = make_llm(sessionmaker, ['{"value": 1}'], price_in_per_mtok=1.0, price_out_per_mtok=2.0)
    await llm.structured(purpose="t", system="s", prompt="q", schema=Answer, use_cache=False)
    [call] = await _calls(sessionmaker)
    assert float(call.cost_usd) == pytest.approx((100 * 1.0 + 20 * 2.0) / 1_000_000)


async def test_rate_limiter_spaces_requests():
    sleeps: list[float] = []

    async def record(seconds):
        sleeps.append(seconds)

    limiter = RateLimiter(rpm=30, sleep=record)  # one request every 2s
    for _ in range(3):
        await limiter.wait()
    # The fake sleep doesn't advance the clock, so requests 2 and 3 are scheduled at
    # +2s and +4s from the first one.
    assert sleeps == [pytest.approx(2, abs=0.05), pytest.approx(4, abs=0.05)]


def test_strip_fences():
    assert json.loads(_strip_fences('```json\n{"a": 1}\n```')) == {"a": 1}
    assert _strip_fences('{"a": 1}') == '{"a": 1}'


async def test_token_budget_refills_continuously():
    from app.llm.client import TokenBudget

    now = [0.0]
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds  # advance the fake clock

    budget = TokenBudget(tpm=6000, sleep=fake_sleep, clock=lambda: now[0])  # 100 tokens/s
    await budget.reserve(4000)
    await budget.reserve(2000)  # bucket now empty, no wait yet
    assert sleeps == []
    await budget.reserve(1000)  # needs 1000 tokens at 100/s → 10 s, not a full minute
    assert sleeps == [pytest.approx(10.0)]

    overhead = TokenBudget.REQUEST_OVERHEAD
    budget.settle(reserved=1000, actual=1000)  # real cost = 1000 + overhead → extra charge
    await budget.reserve(500)
    assert sleeps[-1] == pytest.approx((overhead + 500) / 100)

    before = budget._level
    budget.settle(reserved=2000, actual=100)  # cheaper than reserved → refund the rest
    assert budget._level == pytest.approx(before + 2000 - (100 + overhead))


async def test_token_budget_disabled_never_waits():
    from app.llm.client import TokenBudget

    async def boom(_):
        raise AssertionError("should not sleep")

    budget = TokenBudget(tpm=0, sleep=boom)
    for _ in range(5):
        await budget.reserve(10_000)
