"""
AIOS Configuration — Pydantic Settings loaded from environment variables.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── App ──
    APP_ENV: str = "development"
    APP_VERSION: str = "0.1.0"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000,https://edeziav2.vercel.app,https://ozymorlab.vercel.app"

    # ── Database ──
    DATABASE_URL: str = "postgresql+asyncpg://aios:aios_secret@postgres:5432/aios"
    DATABASE_URL_SYNC: str = "postgresql://aios:aios_secret@postgres:5432/aios"
    DIRECT_URL: str = ""

    # ── Redis ──
    REDIS_URL: str = "redis://redis:6379/0"

    # ── AWS S3 (Storage) ──
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: str = "aios-submissions"

    # ── Google Gemini AI ──
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-pro"

    # ── Grading ──
    GRADING_TEMPERATURE: float = 0.0
    GRADING_MAX_RETRIES: int = 3

    # ── DEIS (Diagram Evaluation Intelligence System) ──
    DEIS_API_URL: str = "http://deis-gateway:8001"
    DEIS_POLL_TIMEOUT: int = 60  # seconds to wait for diagram evaluation
    DEIS_POLL_INTERVAL: int = 2  # seconds between status polls

    # ── Confidence Validation ──
    CONFIDENCE_AUTO_APPROVE: float = 0.6   # Below this → NEEDS_REVIEW
    CONFIDENCE_COMPONENT_FLAG: float = 0.4  # Per-component flag threshold

    # ── Label Validation ──
    LABEL_FUZZY_THRESHOLD: int = 80  # 0-100, minimum similarity score for label match

    # ── Celery ──
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/1"

    # ── JWT Auth ──
    JWT_SECRET_KEY: str = "edexia-secret-change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── Supabase Integration ──
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""  # Service role key to bypass RLS in backend
    SUPABASE_JWT_SECRET: str = ""  # Symmetric verification fallback key
    SUPABASE_STORAGE_BUCKET: str = "aios-submissions"

    # ── Rate Limiting ──
    RATE_LIMIT_DEFAULT: str = "60/minute"
    RATE_LIMIT_AUTH: str = "10/minute"
    RATE_LIMIT_UPLOAD: str = "10/minute"

    @property
    def cors_origins_list(self) -> list[str]:
        origins = [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
        # Always ensure production and new domains are allowed
        for extra in ["https://ozymorlab.vercel.app", "https://edeziav2.vercel.app"]:
            if extra not in origins:
                origins.append(extra)
        
        # Security: Remove localhost/loopback origins in production environment
        if self.APP_ENV.lower() == "production":
            origins = [o for o in origins if "localhost" not in o and "127.0.0.1" not in o]
            
        return origins

    model_config = {"env_file": (".env", "../.env"), "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

# ── PgBouncer Connection String Sanitization ──
# Strip pgbouncer query parameters from the database connection strings.
# This prevents asyncpg and psycopg2 from throwing TypeError: connect() got an unexpected keyword argument 'pgbouncer'
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

def _strip_pgbouncer(url_str: str) -> str:
    if not url_str or "pgbouncer" not in url_str:
        return url_str
    try:
        parsed = urlparse(url_str)
        query_params = parse_qs(parsed.query)
        query_params.pop("pgbouncer", None)
        new_query = urlencode(query_params, doseq=True)
        parsed = parsed._replace(query=new_query)
        return urlunparse(parsed)
    except Exception:
        return url_str

settings.DATABASE_URL = _strip_pgbouncer(settings.DATABASE_URL)
settings.DATABASE_URL_SYNC = _strip_pgbouncer(settings.DATABASE_URL_SYNC)

