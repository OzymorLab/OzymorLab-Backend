"""
AIOS — Assessment Intelligence Operating System
FastAPI application entry point.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.db.session import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Rate Limiter ──
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT_DEFAULT])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("=" * 60)
    logger.info("AIOS Backend starting up...")
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info(f"Version: {settings.APP_VERSION}")
    logger.info(f"Gemini Model: {settings.GEMINI_MODEL}")
    logger.info(f"Rate Limit: {settings.RATE_LIMIT_DEFAULT}")
    logger.info("=" * 60)

    # Initialize database tables
    try:
        await init_db()
        logger.info("Database tables initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    # Ensure S3 bucket exists (if possible with current credentials)
    try:
        from app.services.ingestion import ensure_bucket_exists
        ensure_bucket_exists()
        logger.info(f"S3 bucket '{settings.S3_BUCKET}' ready")
    except Exception as e:
        logger.warning(f"S3 bucket check failed (ensure it exists in AWS): {e}")

    logger.info("AIOS Backend ready!")
    yield
    logger.info("AIOS Backend shutting down...")


# Create FastAPI application
app = FastAPI(
    title="AIOS — Assessment Intelligence Operating System",
    description=(
        "AI-powered assessment grading with observability, drift detection, "
        "and institutional trust for the Indian education ecosystem."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Rate Limiting Middleware ──
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Register API routers
from app.api.health import router as health_router
from app.api.auth import router as auth_router
from app.api.tasks import router as tasks_router
from app.api.submissions import router as submissions_router
from app.api.runs import router as runs_router
from app.api.reviews import router as reviews_router

app.include_router(health_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(tasks_router, prefix="/api/v1")
app.include_router(submissions_router, prefix="/api/v1")
app.include_router(runs_router, prefix="/api/v1")
app.include_router(reviews_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — redirects to docs."""
    return {
        "name": "AIOS — Assessment Intelligence Operating System",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
