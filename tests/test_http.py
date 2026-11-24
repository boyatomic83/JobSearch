import httpx
import pytest

from app.ingest.http import FetchError, PoliteClient


class FakeSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _client(responses: list, sleep: FakeSleep, **kw) -> tuple[PoliteClient, list]:
    """Serve `responses` in order; an Exception item is raised as a transport error."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        item = responses[len(calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    client = PoliteClient(
        transport=httpx.MockTransport(handler), sleep=sleep, min_interval=0, backoff_base=1, **kw
    )
    return client, calls


async def test_retries_429_with_backoff_then_succeeds():
    sleep = FakeSleep()
    responses = [
        httpx.Response(429),
        httpx.ReadTimeout("slow"),
        httpx.Response(200, json={"ok": True}),
    ]
    client, calls = _client(responses, sleep)
    async with client:
        assert await client.get_json("https://api.example.com/x") == {"ok": True}
    assert len(calls) == 3
    assert 1 <= sleep.calls[0] < 2 and 2 <= sleep.calls[1] < 3  # exponential + jitter


async def test_honours_retry_after_header():
    sleep = FakeSleep()
    responses = [httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json=[])]
    client, _ = _client(responses, sleep)
    async with client:
        await client.get_json("https://api.example.com/x")
    assert sleep.calls == [7.0]


async def test_gives_up_after_max_retries():
    client, calls = _client([httpx.Response(503)] * 3, FakeSleep(), max_retries=2)
    async with client:
        with pytest.raises(FetchError, match="after 3 attempts"):
            await client.get_json("https://api.example.com/x")
    assert len(calls) == 3


async def test_does_not_retry_404():
    client, calls = _client([httpx.Response(404)], FakeSleep())
    async with client:
        with pytest.raises(FetchError, match="404"):
            await client.get_json("https://api.example.com/missing")
    assert len(calls) == 1


async def test_min_interval_spaces_requests_to_same_host():
    sleep = FakeSleep()
    responses = [httpx.Response(200, json={})] * 2
    calls: list = []

    def handler(request):
        calls.append(request)
        return responses[len(calls) - 1]

    async with PoliteClient(
        transport=httpx.MockTransport(handler), sleep=sleep, min_interval=5
    ) as client:
        await client.get_json("https://api.example.com/a")
        await client.get_json("https://api.example.com/b")
    assert len(sleep.calls) == 1 and 4 < sleep.calls[0] <= 5
