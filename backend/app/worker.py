"""
Celery application initialization for async task processing.
Handles PDF parsing and LLM grading as background jobs.
"""
from celery import Celery
from app.config import settings

celery_app = Celery(
    "aios",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
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
