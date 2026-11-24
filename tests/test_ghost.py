from datetime import UTC, datetime, timedelta

from app.matching.ghost import GHOST_THRESHOLD, GhostFacts, ghost_score

NOW = datetime(2026, 11, 24, tzinfo=UTC)


def facts(**overrides) -> GhostFacts:
    """A healthy, fresh posting; override fields to add ghost signals."""
    base = dict(
        title="Backend Engineer",
        posted_at=NOW - timedelta(days=3),
        first_seen_at=NOW - timedelta(days=3),
        times_reposted=0,
        description_length=4000,
        has_structure=True,
        has_salary=True,
        has_location=True,
    )
    return GhostFacts(**(base | overrides))


def test_healthy_job_scores_zero():
    assert ghost_score(facts(), NOW) == (0.0, [])


def test_single_soft_signal_never_hides_a_job():
    for f in (
        facts(posted_at=NOW - timedelta(days=200)),
        facts(times_reposted=5),
        facts(description_length=100),
        facts(has_salary=False, has_location=False),
    ):
        score, _ = ghost_score(f, NOW)
        assert score <= GHOST_THRESHOLD


def test_stacked_signals_cross_the_threshold_with_reasons():
    score, reasons = ghost_score(
        facts(
            posted_at=NOW - timedelta(days=120),
            times_reposted=2,
            description_length=300,
            has_salary=False,
        ),
        NOW,
    )
    assert score == 1.0  # 0.35 + 0.30 + 0.30 + 0.05, capped
    assert score > GHOST_THRESHOLD
    assert reasons == [
        "posted 120 days ago (+0.35)",
        "reposted 2x (+0.30)",
        "very short description (300 chars) (+0.30)",
        "no salary listed (+0.05)",
    ]


def test_evergreen_title_and_fallback_to_first_seen():
    score, reasons = ghost_score(
        facts(
            title="Join our Talent Community",
            posted_at=None,
            first_seen_at=NOW - timedelta(days=60),
            has_structure=False,
        ),
        NOW,
    )
    assert score == 0.75  # 0.50 evergreen + 0.15 age (from first_seen_at) + 0.10 structure
    assert reasons[0].startswith("evergreen")
