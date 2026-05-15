"""
Health check API — service status endpoint.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as aioredis

from app.db.session import get_db
from app.config import settings
from app.schemas.common import ApiResponse

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Check the health of all dependent services.
    Returns 200 if the API is running, with individual service statuses.
    """
    status = {
        "api": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "database": "unknown",
        "redis": "unknown",
    }

    # Check PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        status["database"] = "healthy"
    except Exception as e:
        status["database"] = f"unhealthy: {e}"

    # Check Redis
    try:
        r = aioredis.from_url(settings.REDIS_URL)
        await r.ping()
        status["redis"] = "healthy"
        await r.aclose()
    except Exception as e:
        status["redis"] = f"unhealthy: {e}"

    return ApiResponse(data=status)
