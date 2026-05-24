from app.db.session import get_sync_session
from app.db.models import Task, TaskRubric, Submission
from app.services.llm_client import build_alignment_prompt, call_gemini, ALIGNMENT_SYSTEM_PROMPT

session = get_sync_session()

# Get latest task and rubric
task = session.query(Task).order_by(Task.created_at.desc()).first()
rubric = session.query(TaskRubric).filter_by(task_id=task.id, is_active=True).first()

# Get latest submission
sub = session.query(Submission).filter_by(task_id=task.id).order_by(Submission.created_at.desc()).first()

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
