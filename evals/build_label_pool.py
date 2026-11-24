"""Step 1 of building the matching eval set: dump candidates to label (run once).

For each synthetic resume: the retrieval top-20 (so every job that can reach the top-3
is labelled) + up to 8 role-keyword matches *outside* the top-20 that pass the same
filters (so recall@20 has relevant jobs to miss). The pool is labelled by hand into
evals/data/matching_labels.json.

    python -m evals.build_label_pool > /tmp/pool.json
"""

import asyncio
import json
import random

from sqlalchemy import select

from app.db.models import Job
from app.db.session import get_sessionmaker
from app.matching.embeddings import get_embedder
from app.matching.retrieval import build_query
from evals.common import load_resumes, preferences, retrieve

KEYWORDS = {
    "senior_backend_blr": r"backend|platform|distributed|infrastructure|software engineer",
    "ml_engineer_india": r"machine learning|\mml\M|\mai\M|applied scientist|mlops|llm",
    "data_engineer_london": r"data engineer|analytics engineer|data platform|\metl\M",
}


async def main() -> None:
    embedder = get_embedder()
    random.seed(7)
    pool = {}
    async with get_sessionmaker()() as session:
        for resume in load_resumes():
            top = await retrieve(session, embedder, resume, k=20)
            top_ids = {c.job_id for c in top}
            # Same filters as retrieval, but a much larger k, then keep keyword matches.
            prefs = preferences(resume)
            vec = embedder.embed_query("x")
            wide = (await session.execute(build_query(vec, prefs, -1, k=2000))).all()
            extra_ids = [r.id for r in wide if r.id not in top_ids]
            rows = (
                (
                    await session.execute(
                        select(Job.id).where(
                            Job.id.in_(extra_ids),
                            Job.title.regexp_match(KEYWORDS[resume["id"]], flags="i"),
                        )
                    )
                )
                .scalars()
                .all()
            )
            extra = random.sample(list(rows), min(8, len(rows)))
            jobs = {
                j.id: j
                for j in (
                    await session.scalars(select(Job).where(Job.id.in_(list(top_ids) + extra)))
                ).all()
            }
            pool[resume["id"]] = [
                {
                    "key": f"{jobs[i].source}:{jobs[i].source_id}",
                    "in_top20": i in top_ids,
                    "title": jobs[i].title,
                    "company": jobs[i].company,
                    "location": jobs[i].location,
                    "snippet": jobs[i].embedding_text.split("\n\n", 1)[-1][:300],
                }
                for i in [c.job_id for c in top] + extra
            ]
    print(json.dumps(pool, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
