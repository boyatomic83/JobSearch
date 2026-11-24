"""Tailor a resume to one job — using ONLY facts from the original — and prove it.

Pipeline (2 LLM calls):
  1. tailor:  rewrite summary + bullets to fit the job (reorder, rephrase, emphasise).
  2. verify:  a second call fact-checks every claim against the original resume text.
  3. rules:   deterministic backstop — any number, skill or employer that does not
              literally appear in the original is dropped, whatever the LLM said.
Anything flagged by (2) or (3) is removed, and reported back so the user sees what was cut.
"""

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.llm.client import LLMClient


class TailoredRole(BaseModel):
    title: str
    company: str
    dates: str | None = None
    bullets: list[str] = Field(default_factory=list)


class TailoredResume(BaseModel):
    summary: str = Field(description="2-3 sentences aimed at this job")
    skills: list[str] = Field(
        default_factory=list, description="Skills from the resume, most relevant first"
    )
    experience: list[TailoredRole] = Field(default_factory=list)


class Verdict(BaseModel):
    id: str
    supported: bool
    reason: str = ""


class Verification(BaseModel):
    verdicts: list[Verdict]


@dataclass
class Dropped:
    claim: str
    reason: str


@dataclass
class TailoringResult:
    resume: TailoredResume
    dropped: list[Dropped] = field(default_factory=list)


TAILOR_SYSTEM = """You tailor a resume to a specific job posting.
Hard rules:
- Use ONLY facts stated in the original resume. You may reorder, rephrase, merge,
  shorten and emphasise. You may NOT add employers, titles, dates, numbers, tools,
  skills, team sizes, scope or achievements that are not in the original.
- Keep every role's title, company and dates exactly as in the original.
- Put the bullets most relevant to the job first; drop irrelevant ones.
- Only use numbers written in the original. Don't compute totals (e.g. "8 years of
  experience") unless that exact figure appears in the resume.
- skills: only skills written in the original resume, most relevant to the job first.
- summary: 2-3 sentences connecting the candidate's real experience to this job."""

VERIFY_SYSTEM = """You are a strict fact-checker. For each claim, decide whether it is
fully supported by the ORIGINAL resume text.
- Rephrasing and summarising are fine.
- Any new fact makes it unsupported: a number, tool, skill, employer, title, date, team
  size, scope or achievement that the original does not state.
- Return one verdict for EVERY claim id, with a short reason."""

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _claims(resume: TailoredResume) -> dict[str, str]:
    """Flatten into individually checkable claims, keyed by a stable id."""
    claims: dict[str, str] = {}
    for i, sentence in enumerate(s for s in re.split(r"(?<=[.!?])\s+", resume.summary) if s):
        claims[f"summary.{i}"] = sentence
    for i, skill in enumerate(resume.skills):
        claims[f"skill.{i}"] = f"Has the skill: {skill}"
    for r, role in enumerate(resume.experience):
        dates = f" ({role.dates})" if role.dates else ""
        claims[f"role.{r}"] = f"Worked as {role.title} at {role.company}{dates}"
        for b, bullet in enumerate(role.bullets):
            claims[f"role.{r}.bullet.{b}"] = f"At {role.company}: {bullet}"
    return claims


def rule_violations(resume: TailoredResume, original: str) -> dict[str, str]:
    """Deterministic checks the LLM verifier can't talk its way out of."""
    text = _norm(original)
    digits = {n.replace(",", "") for n in _NUMBER.findall(original)}
    bad: dict[str, str] = {}

    def check_numbers(claim_id: str, claim: str) -> None:
        for n in _NUMBER.findall(claim):
            if n.replace(",", "") not in digits:
                bad[claim_id] = f"number {n} is not in the original resume"
                return

    for claim_id, claim in _claims(resume).items():
        if claim_id.startswith("skill."):
            skill = resume.skills[int(claim_id.split(".")[1])]
            if _norm(skill) not in text:
                bad[claim_id] = f"skill {skill!r} is not in the original resume"
        elif claim_id.count(".") == 1 and claim_id.startswith("role."):
            role = resume.experience[int(claim_id.split(".")[1])]
            if _norm(role.company) not in text:
                bad[claim_id] = f"employer {role.company!r} is not in the original resume"
            else:
                check_numbers(claim_id, claim)  # dates
        else:
            # "At <company>:" prefix on bullets has no digits, so only the bullet is checked.
            check_numbers(claim_id, claim)
    return bad


def _apply(resume: TailoredResume, drop: dict[str, str]) -> TailoringResult:
    claims = _claims(resume)
    dropped = [Dropped(claims[cid], reason) for cid, reason in drop.items() if cid in claims]
    summary = " ".join(
        s
        for i, s in enumerate(x for x in re.split(r"(?<=[.!?])\s+", resume.summary) if x)
        if f"summary.{i}" not in drop
    )
    skills = [s for i, s in enumerate(resume.skills) if f"skill.{i}" not in drop]
    roles = []
    for r, role in enumerate(resume.experience):
        if f"role.{r}" in drop:  # an unverified employer/title drops the whole role
            continue
        bullets = [b for i, b in enumerate(role.bullets) if f"role.{r}.bullet.{i}" not in drop]
        roles.append(role.model_copy(update={"bullets": bullets}))
    return TailoringResult(
        TailoredResume(summary=summary, skills=skills, experience=roles), dropped
    )


async def tailor_resume(
    llm: LLMClient,
    *,
    original_text: str,
    job_title: str,
    job_company: str,
    job_text: str,
    user_id: int | None,
) -> TailoringResult:
    draft = await llm.structured(
        purpose="tailor",
        system=TAILOR_SYSTEM,
        prompt=(
            f"ORIGINAL RESUME:\n<<<\n{original_text}\n>>>\n\n"
            f"JOB: {job_title} at {job_company}\n<<<\n{job_text[:1500]}\n>>>"
        ),
        schema=TailoredResume,
        user_id=user_id,
    )

    claims = _claims(draft)
    claim_lines = "\n".join(f"{cid}: {text}" for cid, text in claims.items())
    verification = await llm.structured(
        purpose="tailor_verify",
        system=VERIFY_SYSTEM,
        prompt=f"ORIGINAL RESUME:\n<<<\n{original_text}\n>>>\n\nCLAIMS:\n{claim_lines}",
        schema=Verification,
        user_id=user_id,
    )

    drop: dict[str, str] = {}
    verdicts = {v.id: v for v in verification.verdicts}
    for cid in claims:
        v = verdicts.get(cid)
        if v is None:  # the checker skipped it → we can't vouch for it
            drop[cid] = "not verified"
        elif not v.supported:
            drop[cid] = f"verifier: {v.reason or 'unsupported'}"
    for cid, reason in rule_violations(draft, original_text).items():
        drop.setdefault(cid, f"rule: {reason}")
    return _apply(draft, drop)
