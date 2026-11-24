import math
import os

import pytest

from app.matching import embeddings
from app.matching.embeddings import LocalEmbedder, _normalise
from app.matching.text import (
    MAX_DESCRIPTION_CHARS,
    content_hash,
    find_boilerplate,
    job_embedding_text,
    strip_boilerplate,
)


def test_embedding_text_puts_title_company_first_and_truncates():
    text = job_embedding_text("ML Engineer", "Acme", "x" * 10_000)
    assert text.startswith("ML Engineer\nAcme\n\n")
    assert len(text) == len("ML Engineer\nAcme\n\n") + MAX_DESCRIPTION_CHARS


def test_content_hash_is_stable_and_sensitive():
    assert content_hash("a") == content_hash("a")
    assert content_hash("a") != content_hash("b")


INTRO = "Acme is the leading platform for building rockets and we are hiring fast."
EEO = "Acme is an equal opportunity employer and values diversity of every kind."


def test_find_and_strip_company_boilerplate():
    # Synthetic postings: every one shares the intro + EEO lines, roles differ.
    roles = ["Build Python APIs on Postgres for payments.", "Design Figma flows for mobile.",
             "Run Kubernetes clusters across three regions.", "Own the ML ranking models.",
             "Lead enterprise sales in the Midwest region."]  # fmt: skip
    descs = [f"{INTRO}\n{role}\nRequirements\n{EEO}" for role in roles]
    boiler = find_boilerplate(descs)
    assert boiler == {INTRO, EEO}  # short "Requirements" line is never boilerplate
    assert strip_boilerplate(descs[0], boiler) == f"{roles[0]}\nRequirements"
    assert strip_boilerplate(f"A\n\n{INTRO}\n\n\nB", boiler) == "A\n\nB"  # gaps collapsed


def test_no_boilerplate_judged_for_small_companies():
    assert find_boilerplate([f"{INTRO}\nRole {i} details here." for i in range(4)]) == set()


def test_normalise():
    v = _normalise([3.0, 4.0])
    assert v == [0.6, 0.8]
    assert math.isclose(sum(x * x for x in v), 1.0)


def test_unknown_provider_rejected(monkeypatch):
    embeddings.get_embedder.cache_clear()
    monkeypatch.setattr(embeddings.get_settings(), "embedding_provider", "word2vec")
    with pytest.raises(ValueError, match="unknown EMBEDDING_PROVIDER"):
        embeddings.get_embedder()
    embeddings.get_embedder.cache_clear()


@pytest.mark.skipif(bool(os.environ.get("SKIP_MODEL_TESTS")), reason="SKIP_MODEL_TESTS set")
def test_local_bge_model_is_384d_normalised_and_semantic():
    """Uses the real bge-small model (downloads ~130 MB on first run)."""
    pytest.importorskip("fastembed")
    model = LocalEmbedder("BAAI/bge-small-en-v1.5", cache_dir=".cache/fastembed")
    python_job, chef_job = model.embed_documents(
        [
            "Senior Python Backend Engineer\nAcme\n\nBuild FastAPI services on PostgreSQL.",
            "Head Chef\nBistro\n\nRun the kitchen, design seasonal menus.",
        ]
    )
    query = model.embed_query("Python developer with 5 years of Django and Postgres")
    assert len(query) == 384
    assert math.isclose(sum(x * x for x in query), 1.0, rel_tol=1e-4)

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert dot(query, python_job) > dot(query, chef_job) + 0.1
