import pytest

from app.ingest.clean import (
    extract_usd_salary_range,
    html_to_text,
    looks_remote,
    plausible_annual_salary,
)


def test_html_to_text_keeps_structure_and_inline_text():
    html = "<h2>About</h2><p>We build <b>fast</b> tools.</p><ul><li>Python</li><li>SQL</li></ul>"
    assert html_to_text(html) == "About\nWe build fast tools.\n- Python\n- SQL"


def test_html_to_text_drops_scripts_and_handles_empty():
    assert html_to_text("<p>Hi</p><script>alert(1)</script>") == "Hi"
    assert html_to_text(None) == ""
    assert html_to_text("") == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The base pay range is $150,000 - $200,000 per year.", (150_000, 200_000)),
        ("Salary: $120k–$160k + equity", (120_000, 160_000)),
        ("$95,000 to $125,000 USD", (95_000, 125_000)),
        ("We raised $5B and pay $50-$70/hr", None),  # funding / hourly are ignored
        ("No salary information here", None),
    ],
)
def test_extract_usd_salary_range(text, expected):
    assert extract_usd_salary_range(text) == expected


def test_looks_remote():
    assert looks_remote("Remote - Germany")
    assert looks_remote(None, "Hybrid", "Anywhere in India")
    assert not looks_remote("Bangalore", "Hybrid - London")
    assert not looks_remote("Remotely Park, CA")  # word boundary, not substring


@pytest.mark.parametrize(
    ("low", "high", "currency", "ok"),
    [
        (150_000, 200_000, "USD", True),
        (2_500_000, 4_000_000, "INR", True),
        (None, None, None, True),
        (54, 60, "USD", False),  # hourly intern rate
        (8_000, 8_600, "CAD", False),  # monthly stipend
        (152_405, 179_300_152, "USD", False),  # typo
        (311, 325_000, "USD", False),
        (200_000, 100_000, "USD", False),
    ],
)
def test_plausible_annual_salary(low, high, currency, ok):
    assert plausible_annual_salary(low, high, currency) is ok


def test_job_posting_drops_implausible_salary():
    from app.ingest.base import JobPosting

    job = JobPosting(
        source="t", source_id="1", company="A", title="Intern", url="https://x",
        salary_min=54, salary_max=60, salary_currency="USD",
    )  # fmt: skip
    assert (job.salary_min, job.salary_max, job.salary_currency) == (None, None, None)
