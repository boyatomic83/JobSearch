from app.config import to_async_url


def test_neon_style_url_is_converted_for_asyncpg():
    url = "postgresql://u:p@ep-x.neon.tech/db?sslmode=require&channel_binding=require"
    assert to_async_url(url) == "postgresql+asyncpg://u:p@ep-x.neon.tech/db?ssl=require"


def test_plain_local_url():
    assert to_async_url("postgres://u:p@localhost:5432/db") == (
        "postgresql+asyncpg://u:p@localhost:5432/db"
    )


def test_already_async_url_untouched():
    url = "postgresql+asyncpg://u:p@localhost/db"
    assert to_async_url(url) == url


def test_sslmode_disable_drops_ssl():
    assert to_async_url("postgresql://u:p@h/db?sslmode=disable") == "postgresql+asyncpg://u:p@h/db"


def test_webhook_url_must_be_https_or_empty():
    import pytest
    from pydantic import ValidationError

    from app.config import Settings

    assert Settings(telegram_webhook_url="").telegram_webhook_url == ""
    assert Settings(telegram_webhook_url="https://x.onrender.com").telegram_webhook_url
    with pytest.raises(ValidationError, match="https://"):
        Settings(telegram_webhook_url="# set in production only")


def test_webhook_base_falls_back_to_render_url():
    from app.config import Settings

    s = Settings(telegram_webhook_url="", render_external_url="https://jobscout.onrender.com")
    assert s.webhook_base_url == "https://jobscout.onrender.com"
    assert Settings(telegram_webhook_url="", render_external_url="").webhook_base_url == ""
