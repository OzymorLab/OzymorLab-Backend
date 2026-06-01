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

    # Ensure Supabase Storage bucket is ready
    try:
        from app.services.ingestion import ensure_bucket_exists
        ensure_bucket_exists()
        logger.info(f"Supabase Storage bucket '{settings.SUPABASE_STORAGE_BUCKET}' configured")
    except Exception as e:
        logger.warning(f"Supabase Storage bucket check failed: {e}")

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
origins = settings.cors_origins_list
allow_credentials = True
if "*" in origins or (len(origins) == 1 and origins[0] == "*"):
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global exception handler — ensures CORS headers on 500 errors ──
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all: logs the exception and returns a 500 WITH CORS headers.
    Without this, unhandled exceptions (e.g. Celery broker down) can bypass
    CORSMiddleware in ASGI edge cases, causing browsers to see a CORS error
    instead of the real server error.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}:\n"
        + traceback.format_exc()
    )
    origin = request.headers.get("origin", "")
    response = JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs for details."},
    )
    # Manually attach CORS headers so the browser can read the error body
    allowed_origins = settings.cors_origins_list
    if origin and ("*" in allowed_origins or origin in allowed_origins):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    elif "*" in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Register API routers
from app.api.health import router as health_router
from app.api.auth import router as auth_router
from app.api.tasks import router as tasks_router
from app.api.submissions import router as submissions_router
from app.api.runs import router as runs_router
from app.api.reviews import router as reviews_router
from app.api.question_papers import router as question_papers_router
from app.api.exam_cycles import router as exam_cycles_router
from app.api.schools import router as schools_router
from app.api.reports import router as reports_router
from app.api.analysis import router as analysis_router
from app.api.classroom import router as classroom_router

app.include_router(health_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(tasks_router, prefix="/api/v1")
app.include_router(submissions_router, prefix="/api/v1")
app.include_router(runs_router, prefix="/api/v1")
app.include_router(reviews_router, prefix="/api/v1")
app.include_router(question_papers_router, prefix="/api/v1")
app.include_router(exam_cycles_router, prefix="/api/v1")
app.include_router(schools_router, prefix="/api/v1")
app.include_router(reports_router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")
app.include_router(classroom_router, prefix="/api/v1")



@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — redirects to docs."""
    return {
        "name": "AIOS — Assessment Intelligence Operating System",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
