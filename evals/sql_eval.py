"""Text-to-SQL eval: answer accuracy on 15 questions + guardrails vs malicious input.

    python -m evals.sql_eval

Accuracy: the LLM's SQL (through the real guarded pipeline) must return the same result
as hand-written gold SQL, compared as sets of values (column names and row order may
differ, e.g. for "top 5 companies").
Guardrails: 5 malicious SQL strings must be rejected before execution, and 3 prompt
injections through the natural-language path must not leak or change anything.
"""

import asyncio
import itertools
import json

from sqlalchemy import text

from app.agent.sql_guard import UnsafeSQL
from app.agent.text_to_sql import answer_question, run_readonly
from app.db.session import get_engine
from app.llm.client import get_llm_client
from evals.common import EVALS

DATA = EVALS / "data" / "sql_questions.json"
RESULTS = EVALS / "results" / "sql.json"


def normalise(rows: list[dict]) -> set[tuple]:
    def value(v):
        return round(float(v), 2) if isinstance(v, int | float) else v

    return {tuple(value(v) for v in row.values()) for row in rows}


def matches(got: set[tuple] | None, gold: set[tuple]) -> bool:
    """Equal, or `got` has extra columns (e.g. a count next to "top 5 companies") and some
    choice of its columns reproduces the gold answer exactly."""
    if got is None:
        return False
    if got == gold:
        return True
    width = len(next(iter(gold), ()))
    got_width = len(next(iter(got), ()))
    if not gold or got_width <= width:
        return False
    return any(
        {tuple(row[i] for i in cols) for row in got} == gold
        for cols in itertools.combinations(range(got_width), width)
    )


async def accuracy(engine, llm, questions: list[dict]) -> list[dict]:
    out = []
    for item in questions:
        gold = normalise(await run_readonly(engine, item["gold"]))
        answer = await answer_question(llm, engine, item["q"])
        got = normalise(answer["rows"]) if "rows" in answer else None
        out.append(
            {
                "q": item["q"],
                "correct": matches(got, gold),
                "sql": answer.get("sql"),
                "error": answer.get("error"),
                "gold": sorted(map(str, gold))[:5],
                "got": sorted(map(str, got))[:5] if got is not None else None,
            }
        )
    return out


async def guardrails(engine, llm, data: dict) -> list[dict]:
    out = []
    for item in data["malicious_sql"]:
        try:
            await run_readonly(engine, item["sql"])
            blocked, why = False, "executed!"
        except UnsafeSQL as exc:
            blocked, why = True, str(exc)
        out.append({"input": item["name"], "blocked": blocked, "detail": why})

    async with engine.connect() as conn:
        jobs_before = (await conn.execute(text("SELECT count(*) FROM jobs"))).scalar_one()
    for prompt in data["prompt_injections"]:
        answer = await answer_question(llm, engine, prompt)
        leaked = "raw_text" in json.dumps(answer.get("rows", []))
        out.append(
            {
                "input": f"NL: {prompt[:50]}",
                "blocked": not leaked,
                "detail": answer.get("error") or f"ran harmless: {answer.get('sql')}",
            }
        )
    async with engine.connect() as conn:
        jobs_after = (await conn.execute(text("SELECT count(*) FROM jobs"))).scalar_one()
    out.append(
        {
            "input": "jobs table intact after all attacks",
            "blocked": jobs_after == jobs_before,
            "detail": f"{jobs_before} -> {jobs_after}",
        }
    )
    return out


async def main() -> None:
    data = json.loads(DATA.read_text())
    engine, llm = get_engine(), get_llm_client()
    try:
        acc = await accuracy(engine, llm, data["questions"])
        guard = await guardrails(engine, llm, data)
    finally:
        await engine.dispose()
    correct = sum(r["correct"] for r in acc)
    result = {
        "accuracy": round(correct / len(acc), 3),
        "correct": correct,
        "total": len(acc),
        "guardrails_passed": sum(g["blocked"] for g in guard),
        "guardrails_total": len(guard),
        "questions": acc,
        "guardrails": guard,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=1, default=str))
    print(f"Accuracy: {correct}/{len(acc)} = {result['accuracy']:.0%}")
    for r in acc:
        if not r["correct"]:
            print(f"  ✗ {r['q']}\n     got {r['got']} expected {r['gold']}\n     sql: {r['sql']}")
    print(f"Guardrails: {result['guardrails_passed']}/{len(guard)} held")
    for g in guard:
        print(f"  {'✓' if g['blocked'] else '✗'} {g['input']}: {str(g['detail'])[:110]}")


if __name__ == "__main__":
    asyncio.run(main())
