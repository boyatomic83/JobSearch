"""Resume upload: PDF → text → LLM profile → embedding → stored as the active resume."""

import asyncio

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Resume
from app.db.queries import get_or_create_preferences
from app.llm.client import LLMClient
from app.matching.embeddings import Embedder
from app.resume.parse import extract_profile, extract_text
from app.resume.schemas import ResumeProfile


async def save_resume(
    sessionmaker: async_sessionmaker[AsyncSession],
    llm: LLMClient,
    embedder: Embedder,
    *,
    user_id: int,
    file_name: str | None,
    pdf_bytes: bytes,
) -> tuple[Resume, ResumeProfile]:
    """Raises ResumeError (bad PDF) or LLMError (extraction failed)."""
    text = extract_text(pdf_bytes)
    # The slow LLM call happens before we open a transaction, so no DB connection is held.
    profile = await extract_profile(llm, text, user_id)

    async with sessionmaker() as session, session.begin():
        prefs = await get_or_create_preferences(session, user_id)
        vector = await asyncio.to_thread(embedder.embed_query, profile.embedding_text(prefs.roles))
        # One active resume per user: the new upload replaces the old one.
        await session.execute(
            update(Resume)
            .where(Resume.user_id == user_id, Resume.is_active)
            .values(is_active=False)
        )
        resume = Resume(
            user_id=user_id,
            file_name=file_name,
            raw_text=text,
            profile=profile.model_dump(mode="json"),
            embedding=vector,
        )
        session.add(resume)
    return resume, profile
