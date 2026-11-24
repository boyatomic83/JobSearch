"""Matching eval: precision@3 and recall@20, retrieval-only vs retrieval + LLM rerank.

    python -m evals.matching_eval

Labels: evals/data/matching_labels.json (synthetic resumes x real jobs, hand-labelled).
Nothing is written to `matches` (eval user id -1), so it can run against any database.
Jobs that closed since labelling are skipped; unlabelled jobs reaching the top-k count
as not relevant (pessimistic) and are reported.
"""

import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from app.db.models import Job
from app.db.session import get_engine, get_sessionmaker
from app.llm.client import get_llm_client
from app.matching.embeddings import get_embedder
from app.matching.rerank import rerank
from evals.common import EVALS, load_resumes, preferences, profile, retrieve

LABELS = EVALS / "data" / "matching_labels.json"
RESULTS = EVALS / "results" / "matching.json"


def precision_at(k: int, ranked: list[int], relevant: set[int]) -> float:
    return sum(j in relevant for j in ranked[:k]) / k


def recall_at(k: int, ranked: list[int], relevant: set[int]) -> float | None:
    return sum(j in relevant for j in ranked[:k]) / len(relevant) if relevant else None


async def evaluate() -> dict:
    labels = json.loads(LABELS.read_text())["resumes"]
    embedder, llm = get_embedder(), get_llm_client()
    rows = []
    async with get_sessionmaker()() as session:
        jobs = (await session.execute(select(Job.id, Job.source, Job.source_id))).all()
        id_of = {f"{j.source}:{j.source_id}": j.id for j in jobs}
        for resume in load_resumes():
            labelled = labels[resume["id"]]
            known = {id_of[x["key"]]: x["relevant"] for x in labelled if x["key"] in id_of}
            relevant = {jid for jid, rel in known.items() if rel}
            # Own short transaction: retrieval uses SET LOCAL for the HNSW scan settings.
            async with get_sessionmaker()() as tx, tx.begin():
                top20 = await retrieve(tx, embedder, resume, k=20)
            retrieval_ids = [c.job_id for c in top20]
            reranked = await rerank(
                llm, profile(resume), preferences(resume), top20, user_id=None, n=3
            )
            rerank_ids = [c.job_id for c, _ in reranked]
            rows.append(
                {
                    "resume": resume["id"],
                    "labelled_pairs": len(labelled),
                    "missing_jobs": len(labelled) - len(known),
                    "unlabelled_in_top20": sum(j not in known for j in retrieval_ids),
                    "relevant": len(relevant),
                    "p@3_retrieval": precision_at(3, retrieval_ids, relevant),
                    "p@3_rerank": precision_at(3, rerank_ids, relevant),
                    # Best any reranker could do with this top-20 (an upper bound for context).
                    "p@3_oracle": min(3, sum(j in relevant for j in retrieval_ids)) / 3,
                    "r@20": recall_at(20, retrieval_ids, relevant),
                    "rerank_fell_back": any(k is None for _, k in reranked),
                }
            )
    keys = ["p@3_retrieval", "p@3_rerank", "p@3_oracle", "r@20"]
    mean = {
        k: round(
            sum(r[k] for r in rows if r[k] is not None)
            / max(1, sum(r[k] is not None for r in rows)),
            3,
        )
        for k in keys
    }
    return {"per_resume": rows, "mean": mean}


def table(result: dict) -> str:
    lines = [
        "| resume | relevant | P@3 retrieval | P@3 +rerank | P@3 oracle | R@20 |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["per_resume"]:
        lines.append(
            f"| {r['resume']} | {r['relevant']} | {r['p@3_retrieval']:.2f} | "
            f"{r['p@3_rerank']:.2f} | {r['p@3_oracle']:.2f} | {r['r@20']:.2f} |"
        )
    m = result["mean"]
    lines.append(
        f"| **mean** | | **{m['p@3_retrieval']:.2f}** | **{m['p@3_rerank']:.2f}** | "
        f"{m['p@3_oracle']:.2f} | **{m['r@20']:.2f}** |"
    )
    return "\n".join(lines)


async def main() -> None:
    try:
        result = await evaluate()
    finally:
        await get_engine().dispose()
    Path(RESULTS).parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=1))
    print(table(result))
    print("\nR@20 is the same with and without rerank: rerank only reorders the top 20.")
    for r in result["per_resume"]:
        if r["missing_jobs"] or r["unlabelled_in_top20"] or r["rerank_fell_back"]:
            print(
                f"note {r['resume']}: {r['missing_jobs']} labelled jobs closed, "
                f"{r['unlabelled_in_top20']} unlabelled in top-20, "
                f"rerank fallback={r['rerank_fell_back']}"
            )


if __name__ == "__main__":
    asyncio.run(main())
