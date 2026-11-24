"""CLI: `python -m app.ingest.run [--source greenhouse lever remoteok]`."""

import argparse
import asyncio
import json
import logging

from app.db.session import get_engine, get_sessionmaker
from app.ingest.service import CONNECTORS, build_connectors, run_ingest


async def main(sources: list[str] | None) -> None:
    try:
        report = await run_ingest(get_sessionmaker(), build_connectors(sources))
    finally:
        await get_engine().dispose()
    print(json.dumps(report.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest jobs from public job boards.")
    parser.add_argument("--source", nargs="*", choices=sorted(CONNECTORS), help="default: all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main(args.source))
