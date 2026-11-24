"""Small shared lookups used by the API, matching and (later) the agent and bot."""

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Preference, Resume, User


async def get_user(session: AsyncSession, user_id: int) -> User | None:
    return await session.get(User, user_id)


async def get_or_create_preferences(session: AsyncSession, user_id: int) -> Preference:
    """Every user has exactly one preferences row; create it lazily (race-safe)."""
    await session.execute(
        insert(Preference)
        .values(user_id=user_id)
        .on_conflict_do_nothing(index_elements=["user_id"])
    )
    return (await session.scalars(select(Preference).where(Preference.user_id == user_id))).one()


async def get_active_resume(session: AsyncSession, user_id: int) -> Resume | None:
    return await session.scalar(
        select(Resume)
        .where(Resume.user_id == user_id, Resume.is_active)
        .order_by(Resume.created_at.desc())
        .limit(1)
    )
