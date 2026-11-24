"""Paid-tier list prices, used only to *estimate* cost (the free tier bills $0).

Source: https://console.groq.com/docs/models ("price per 1M tokens"), fetched 2026-11-24.
Update this table if prices change; LLM_PRICE_IN/OUT_PER_MTOK env vars override it.
"""

# model -> (USD per 1M input tokens, USD per 1M output tokens)
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}


def prices_for(
    model: str, override_in: float = 0.0, override_out: float = 0.0
) -> tuple[float, float]:
    if override_in or override_out:
        return override_in, override_out
    return MODEL_PRICES.get(model, (0.0, 0.0))
