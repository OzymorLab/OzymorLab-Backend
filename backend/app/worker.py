"""
Celery application initialization for async task processing.
Handles PDF parsing and LLM grading as background jobs.
"""
from celery import Celery
from app.config import settings

from urllib.parse import urlparse, urlunparse

# Upstash Redis only supports database 0. Dynamically rewrite database index from /1 to /0 to avoid Kombu OperationalError.
def sanitize_redis_url(url_str: str) -> str:
    if not url_str:
        return url_str
    if "upstash.io" in url_str:
        try:
            parsed = urlparse(url_str)
            if parsed.path and parsed.path != "/0":
                parsed = parsed._replace(path="/0")
            return urlunparse(parsed)
        except Exception:
            pass
    return url_str

broker_url = sanitize_redis_url(settings.celery_broker)
backend_url = sanitize_redis_url(settings.celery_backend)

celery_app = Celery(
    "aios",
    broker=broker_url,
    backend=backend_url,
)

# Resilient SSL certificate configuration for secure Upstash Redis (rediss://)
is_secure_broker = broker_url.startswith("rediss://")
is_secure_backend = backend_url.startswith("rediss://")

celery_app.conf.update(
    broker_use_ssl={"ssl_cert_reqs": 0} if is_secure_broker else False,
    redis_backend_use_ssl={"ssl_cert_reqs": 0} if is_secure_backend else False,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Route tasks to specific queues if needed later
    task_routes={
        "app.tasks.parse_submission.*": {"queue": "parsing"},
        "app.tasks.grade_submission.*": {"queue": "grading"},
    },
    # Default queue for unrouted tasks
    task_default_queue="default",
    imports=[
        "app.tasks.parse_submission",
        "app.tasks.grade_submission",
        "app.tasks.orchestrator",
        "app.tasks.extract_identity",
    ],
)

# Auto-discover task modules
celery_app.autodiscover_tasks(["app.tasks"])
