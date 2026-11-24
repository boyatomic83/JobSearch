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
