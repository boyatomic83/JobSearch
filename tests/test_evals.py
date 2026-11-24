"""The eval scripts' own logic (metrics, result comparison, label file integrity)."""

import json

from evals.common import EVALS, load_resumes
from evals.matching_eval import precision_at, recall_at
from evals.sql_eval import matches


def test_precision_and_recall():
    assert precision_at(3, [1, 2, 3, 4], {1, 3}) == 2 / 3
    assert recall_at(20, [1, 2], {1, 9}) == 0.5
    assert recall_at(20, [1], set()) is None


def test_sql_result_matching_allows_extra_columns_only():
    assert matches({("A", 631.0), ("B", 5.0)}, {("A",), ("B",)})
    assert not matches({("A", 631.0)}, {("A",), ("B",)})
    assert not matches({(6.0,)}, {(5.0,)}) and not matches(None, {(1.0,)})


def test_eval_data_is_consistent():
    resumes = load_resumes()
    labels = json.loads((EVALS / "data" / "matching_labels.json").read_text())["resumes"]
    assert {r["id"] for r in resumes} == set(labels)
    assert all("SYNTHETIC" in r["_note"] for r in resumes)
    pairs = sum(len(v) for v in labels.values())
    assert 50 <= pairs <= 100
    sql = json.loads((EVALS / "data" / "sql_questions.json").read_text())
    assert len(sql["questions"]) == 15 and len(sql["malicious_sql"]) == 5
