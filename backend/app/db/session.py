"""
Database session management — async SQLAlchemy engine and session factory.
"""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

from uuid import uuid4

# Configure transaction pooler (PgBouncer) prepared statement safety.
# We disable prepared statements completely for both SQLAlchemy and asyncpg to ensure PgBouncer compatibility.
connect_args = {
    "statement_cache_size": 0,
    "prepared_statement_cache_size": 0,
    "prepared_statement_name_func": lambda: f"__asyncpg_{uuid4()}__",
}

# Async engine for FastAPI
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_ENV == "development",
    connect_args=connect_args,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)



# Async session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields an async DB session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Create all tables (used on startup if Alembic is not yet running)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed default math question steps and roster
    from app.db.seed import seed_database
    async with async_session_factory() as session:
        try:
            await seed_database(session)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Failed to auto-seed database: {e}")
