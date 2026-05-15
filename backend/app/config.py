"""
AIOS Configuration — Pydantic Settings loaded from environment variables.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── App ──
    APP_ENV: str = "development"
    APP_VERSION: str = "0.1.0"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    # ── Database ──
    DATABASE_URL: str = "postgresql+asyncpg://aios:aios_secret@postgres:5432/aios"
    DATABASE_URL_SYNC: str = "postgresql://aios:aios_secret@postgres:5432/aios"

    # ── Redis ──
    REDIS_URL: str = "redis://redis:6379/0"

    # ── MinIO ──
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_USER: str = "minioadmin"
    MINIO_PASSWORD: str = "minioadmin123"
    MINIO_BUCKET: str = "aios-submissions"
    MINIO_SECURE: bool = False

    # ── Google Gemini AI ──
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-pro"

    # ── Grading ──
    GRADING_TEMPERATURE: float = 0.0
    GRADING_MAX_RETRIES: int = 3

    # ── Celery ──
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/1"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
