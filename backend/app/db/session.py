"""
Database session management — async SQLAlchemy engine and session factory.
"""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

# Configure transaction pooler (PgBouncer) prepared statement safety.
# We disable prepared statements completely for both SQLAlchemy and asyncpg to ensure PgBouncer compatibility.
connect_args = {
    "statement_cache_size": 0
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


# Sync engine for Celery workers
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sync_engine = create_engine(
    settings.DATABASE_URL_SYNC,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)
sync_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

def get_sync_session():
    """Returns a new synchronous Session from the shared engine."""
    return sync_session_factory()


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
