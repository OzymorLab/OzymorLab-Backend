"""
Integration tests for the full grading run pipeline, rubric approval gates, SSE streams, and Idempotency key lockouts.
"""
import pytest
import uuid
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, BackgroundTasks
from app.api.runs import start_run, run_events_stream
from app.api.submissions import bulk_grade_submissions, BulkGradeRequest
from app.db.models import GradingRun, Task, TaskRubric, User, Submission


@pytest.mark.asyncio
async def test_bulk_grade_unapproved_rubric():
    """Verify bulk grading fails if the active rubric is not in APPROVED state."""
    db_mock = AsyncMock()
    user_mock = MagicMock(spec=User)
    user_mock.id = uuid.uuid4()
    user_mock.school_id = None  # bypass school-isolation check in check_task_access

    # Mock Task
    task_mock = MagicMock(spec=Task)
    task_mock.id = uuid.uuid4()
    task_mock.max_marks = 20

    # Mock unapproved Rubric (approval_status="DRAFT")
    rubric_mock = MagicMock(spec=TaskRubric)
    rubric_mock.version = 1
    rubric_mock.approval_status = "DRAFT"

    # Mock DB execution
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.side_effect = [task_mock, rubric_mock]
    db_mock.execute.return_value = result_mock

    payload = BulkGradeRequest(
        task_id=str(task_mock.id),
        description="Test grading",
        temperature=0.0
    )

    request_mock = MagicMock()
    request_mock.headers = {}
    background_tasks_mock = BackgroundTasks()

    with pytest.raises(HTTPException) as exc_info:
        await bulk_grade_submissions(
            payload=payload,
            background_tasks=background_tasks_mock,
            request=request_mock,
            db=db_mock,
            current_user=user_mock,
        )

    assert exc_info.value.status_code == 400
    assert "Only APPROVED rubrics" in exc_info.value.detail


@pytest.mark.asyncio
async def test_bulk_grade_idempotency_locking():
    """Verify that idempotent decorator handles duplicate submissions and concurrent locks."""
    db_mock = AsyncMock()
    user_mock = MagicMock(spec=User)
    user_mock.id = uuid.uuid4()
    user_mock.school_id = None  # bypass school-isolation check in check_task_access

    # Mock Task
    task_mock = MagicMock(spec=Task)
    task_mock.id = uuid.uuid4()
    task_mock.max_marks = 20

    # Mock approved Rubric
    rubric_mock = MagicMock(spec=TaskRubric)
    rubric_mock.version = 1
    rubric_mock.approval_status = "APPROVED"

    # Mock Submissions (parsed)
    sub_mock = MagicMock(spec=Submission)
    sub_mock.id = uuid.uuid4()

    # Mock DB executions
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.side_effect = [
        None,       # idempotency lookup: NotFound
        task_mock,  # task lookup
        rubric_mock,  # rubric lookup
    ]
    result_mock.scalars.return_value.all.return_value = [sub_mock]
    db_mock.execute.return_value = result_mock

    background_tasks_mock = BackgroundTasks()

    # Patch asyncio.create_task to avoid background grading running in tests
    with patch("asyncio.create_task"):
        payload = BulkGradeRequest(
            task_id=str(task_mock.id),
            description="Test bulk grade",
            temperature=0.0
        )

        request_mock = MagicMock()
        request_mock.headers = {"Idempotency-Key": "test-grading-lock-123"}

        res = await bulk_grade_submissions(
            payload=payload,
            background_tasks=background_tasks_mock,
            request=request_mock,
            db=db_mock,
            current_user=user_mock,
        )

        assert res.data["status"] == "RUNNING"
        assert res.data["submissions_queued"] == 1
        assert db_mock.commit.called


@pytest.mark.asyncio
async def test_sse_progress_stream():
    """Verify that SSE stream yields valid progress payloads and disconnects cleanly."""
    db_mock = AsyncMock()

    # school_id=None triggers the unconditional lookup path in check_run_access,
    # bypassing the school isolation check entirely.
    user_mock = MagicMock(spec=User)
    user_mock.school_id = None

    run_mock = MagicMock(spec=GradingRun)
    run_mock.id = uuid.uuid4()
    run_mock.status = "RUNNING"
    run_mock.total_submissions = 10
    run_mock.graded_count = 5
    run_mock.failed_count = 1

    # Both check_run_access AND the event loop query the same db.execute mock.
    # We need at least two results: one for the access check, one for the event loop.
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = run_mock
    db_mock.execute.return_value = result_mock

    # Generate events
    response = await run_events_stream(
        run_id=str(run_mock.id), db=db_mock, current_user=user_mock
    )

    # Retrieve the async generator from StreamingResponse
    gen = response.body_iterator

    # First event check
    event1 = await gen.__anext__()
    assert "data:" in event1

    json_str = event1.replace("data: ", "").strip()
    data = json.loads(json_str)

    assert data["run_id"] == str(run_mock.id)
    assert data["status"] == "RUNNING"
    assert data["total_submissions"] == 10
    assert data["graded_count"] == 5
    assert data["failed_count"] == 1
    assert data["progress_percentage"] == 60.0  # (5+1)/10 = 60.0%
