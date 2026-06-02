"""
Debug script: test alignment prompt against the latest task/submission.
Run with: python debug_alignment.py (from the backend/ directory)
"""
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import async_session_factory
from app.db.models import Task, TaskRubric, Submission
from app.services.llm_client import build_alignment_prompt, call_gemini, ALIGNMENT_SYSTEM_PROMPT
from sqlalchemy import select


async def main():
    async with async_session_factory() as session:
        # Get latest task and rubric
        task_res = await session.execute(
            select(Task).order_by(Task.created_at.desc()).limit(1)
        )
        task = task_res.scalar_one_or_none()

        rubric_res = await session.execute(
            select(TaskRubric).filter_by(task_id=task.id, is_active=True).limit(1)
        )
        rubric = rubric_res.scalar_one_or_none()

        # Get latest submission
        sub_res = await session.execute(
            select(Submission)
            .filter_by(task_id=task.id)
            .order_by(Submission.created_at.desc())
            .limit(1)
        )
        sub = sub_res.scalar_one_or_none()

    print(f"Task: {task.title}")
    print(f"Submission: {sub.file_name}")

    r_steps = rubric.rubric_json["steps"]
    s_steps = sub.parsed_content["steps"]

    prompt = build_alignment_prompt(r_steps, s_steps)
    print("\n--- Alignment Prompt ---")
    print(prompt)

    result = call_gemini(
        prompt, system_prompt=ALIGNMENT_SYSTEM_PROMPT, call_type="alignment"
    )
    print("\n--- Success ---")
    print(result["success"])
    print("\n--- Response Text ---")
    print(result["response_text"])


if __name__ == "__main__":
    asyncio.run(main())
