"""CLI: `python -m app.matching.run [--reembed]` — embed, dedup and ghost-score jobs."""

import argparse
import asyncio
import json
import logging

from sqlalchemy import update

from app.db.models import Job
from app.db.session import get_engine, get_sessionmaker
from app.matching.embeddings import get_embedder
from app.matching.process import run_processing


async def main(reembed: bool) -> None:
    sessionmaker = get_sessionmaker()
    try:
        if reembed:  # e.g. after switching EMBEDDING_PROVIDER / EMBEDDING_MODEL
            async with sessionmaker() as session, session.begin():
                await session.execute(update(Job).values(embedding=None))
        report = await run_processing(sessionmaker, get_embedder())
    finally:
        await get_engine().dispose()
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed, dedup and ghost-score jobs.")
    parser.add_argument("--reembed", action="store_true", help="re-embed every job")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main(args.reembed))
