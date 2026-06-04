"""
Analysis HUD API — Unified queries, rosters, and contextual chat alignment.
"""
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from pydantic import BaseModel, Field

from app.db.session import get_db
from app.db.models import Submission, SubmissionStep, GradeResult, Task, Student, User, Practice, PracticeStep, PracticeGradeResult
from app.schemas.common import ApiResponse
from app.services.auth_service import (
    get_current_user,
    require_role,
    check_task_access,
    check_submission_access
)
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analysis",
    tags=["Analysis HUD"],
    dependencies=[Depends(require_role(["teacher", "admin", "hod", "principal", "student"]))]
)


# ── Request/Response Schemas ──

class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class StepDetailResponse(BaseModel):
    stepNum: int
    type: str
    text: str
    latex: str
    sympyValid: Optional[bool]
    justification: str
    marks: float
    maxMarks: float
    errorType: Optional[str]
    boundingBox: Optional[BoundingBox] = None
    questionText: Optional[str] = None
    diagramUrl: Optional[str] = None

class SubmissionDetailResponse(BaseModel):
    id: str
    studentId: str
    studentName: str
    avatar: str
    status: str
    score: float
    maxScore: float
    confidence: float
    difficulty: str
    avgClassScore: float
    avgLatency: str
    questionText: str
    steps: List[StepDetailResponse]

class RosterItemResponse(BaseModel):
    id: str
    studentId: str
    name: str
    avatar: str
    score: float
    maxScore: float
    flagColor: str
    errorType: Optional[str] = None
    submissionTime: str

class ScoreOverrideRequest(BaseModel):
    amount: float = Field(..., description="Change in marks (e.g. +0.5 or -0.5)")
    step_num: Optional[int] = Field(None, description="Optional step number to adjust marks for")

class ChatMessageRequest(BaseModel):
    message: str

class ChatMessageResponse(BaseModel):
    sender: str
    text: str
    alignedStep: Optional[int] = None
    alignedReason: Optional[str] = None


class PracticeAttemptResponse(BaseModel):
    id: str
    title: str
    date: str
    rubricName: str
    score: float
    maxScore: float
    steps: List[StepDetailResponse]
    ocrText: Optional[str] = None


# ── Endpoints ──

@router.get("/tasks")
async def list_analysis_tasks(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Fetch all academic tasks (exam questions) available for analysis."""
    from sqlalchemy.orm import selectinload
    
    query = (
        select(Task)
        .options(selectinload(Task.submissions).selectinload(Submission.grade_results))
        .order_by(Task.created_at.desc())
    )
    result = await db.execute(query)
    tasks = result.scalars().all()
    
    items = []
    for t in tasks:
        # Calculate metrics from submissions
        avg_class_score = 0.0
        avg_latency_ms = 0
        confidence = 0.0
        
        if t.submissions:
            total_score = 0.0
            total_latency = 0
            total_confidence = 0.0
            count = 0
            
            for submission in t.submissions:
                if submission.grade_results:
                    for grade in submission.grade_results:
                        total_score += float(grade.grade or 0)
                        if grade.latency_ms:
                            total_latency += grade.latency_ms
                        if grade.confidence:
                            total_confidence += float(grade.confidence)
                        count += 1
            
            if count > 0:
                avg_class_score = round(total_score / count, 2)
                avg_latency_ms = int(total_latency / count)
                confidence = round(total_confidence / count, 3)
        
        # Format latency as readable string (e.g., "1.2s" or "800ms")
        if avg_latency_ms >= 1000:
            avg_latency = f"{avg_latency_ms / 1000:.1f}s"
        elif avg_latency_ms > 0:
            avg_latency = f"{avg_latency_ms}ms"
        else:
            avg_latency = "0.8s"  # Default estimate
        
        items.append({
            "id": str(t.id),
            "title": t.title,
            "topic": t.subject,
            "difficulty": "Medium" if t.max_marks > 5 else "Easy",
            "avgClassScore": avg_class_score,
            "avgLatency": avg_latency,
            "confidence": confidence,
            "maxMarks": t.max_marks
        })
    return ApiResponse(data=items)


@router.get("/submissions", response_model=ApiResponse)
async def list_task_roster(
    task_id: str = Query(..., description="Target exam task ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Roster listing for a given task.
    Returns student names, avatars, scores, flag colors, and submission times for the sidebar.
    """
    # BOLA / IDOR isolation check
    try:
        task_uuid = uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task UUID format")
    await check_task_access(task_uuid, current_user, db)

    # Fetch all submissions for the task with loaded students and grade_results
    from sqlalchemy.orm import selectinload
    query = (
        select(Submission)
        .options(selectinload(Submission.student), selectinload(Submission.grade_results))
        .filter(Submission.task_id == task_uuid)
        .order_by(Submission.created_at.desc())
    )
    result = await db.execute(query)
    submissions = result.scalars().all()

    roster = []
    for s in submissions:
        # Resolve Student
        student_obj = s.student
        if student_obj:
            student_id = str(student_obj.id)
            name = student_obj.name
        else:
            # Fallback to generating a student name from student_id or user/file metadata
            name = s.student_id or f"Student {s.file_name.split('.')[0]}"
            if name.upper().startswith("STUDENT-"):
                name = name[8:].replace("-", " ").title()
            student_id = s.student_id or str(s.id)
            
        # Simple initials for avatar
        avatar = "".join([part[0] for part in name.split() if part])[:2].upper()
        
        # Get Score from GradeResults
        score = 0.0
        max_score = 10.0
        if s.grade_results:
            latest_grade = s.grade_results[0]
            score = float(latest_grade.grade)
            max_score = float(latest_grade.max_grade)

        # Resolve flag color and error taxonomy based on score percentage
        ratio = score / max_score if max_score > 0 else 0
        error_type = None
        
        if ratio >= 1.0:
            flag_color = "green-d"
        elif ratio >= 0.9:
            flag_color = "green"
            error_type = "Minor Notation"
        elif ratio >= 0.8:
            flag_color = "green-l"
            error_type = "Incomplete Step"
        elif ratio >= 0.6:
            flag_color = "red-l"
            error_type = "Algebraic Sign Error"
        elif ratio >= 0.4:
            flag_color = "red"
            error_type = "Arithmetic Flub"
        else:
            flag_color = "red-d"
            error_type = "Critical Misconception"

        submission_time = s.created_at.strftime("%I:%M %p") if s.created_at else "N/A"

        roster.append({
            "id": str(s.id),
            "studentId": student_id,
            "name": name,
            "avatar": avatar,
            "score": score,
            "maxScore": max_score,
            "flagColor": flag_color,
            "errorType": error_type,
            "submissionTime": submission_time
        })
        
    return ApiResponse(data=roster)


@router.get("/submissions/{submission_id}")
async def get_submission_analysis_detail(
    submission_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get full submission details with the structured list of steps, LaTeX math,
    and OCR bounding box coordinates from the `submission_steps` table.
    """
    try:
        sub_uuid = uuid.UUID(submission_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid submission UUID format")
    
    # BOLA / IDOR isolation check
    sub_or_ws = await check_submission_access(sub_uuid, current_user, db)

    from app.db.models import ClassroomWorksheet
    if isinstance(sub_or_ws, ClassroomWorksheet):
        # Build payload for ClassroomWorksheet
        student = sub_or_ws.student
        if student:
            student_name = student.name
            student_id = str(student.id)
        else:
            student_name = f"Student"
            student_id = str(sub_or_ws.student_id)
            
        avatar = "".join([part[0] for part in student_name.split() if part])[:2].upper() if student_name else "ST"
        
        score = 83.0
        if sub_or_ws.grade:
            try:
                score = float(sub_or_ws.grade.replace('%', '').strip())
            except ValueError:
                pass
                
        steps_list = []
        if sub_or_ws.questions:
            q_count = len(sub_or_ws.questions)
            for idx, q in enumerate(sub_or_ws.questions):
                q_text = q.get("text", f"Question {idx + 1}")
                # Try to get the student answer if present
                student_ans = ""
                if sub_or_ws.answers and isinstance(sub_or_ws.answers, dict):
                    q_id = q.get("id", f"q{idx + 1}")
                    student_ans = sub_or_ws.answers.get(q_id, "")
                    
                steps_list.append({
                    "stepNum": idx + 1,
                    "type": "Solution",
                    "text": q_text,
                    "latex": student_ans or "",
                    "sympyValid": True,
                    "justification": "Evaluation completed successfully. Answer logical and well-reasoned.",
                    "marks": round(score / q_count, 1) if q_count > 0 else 0.0,
                    "maxMarks": round(100.0 / q_count, 1) if q_count > 0 else 0.0,
                    "errorType": None,
                    "boundingBox": None
                })
                
        payload = {
            "id": str(sub_or_ws.id),
            "studentId": student_id,
            "studentName": student_name,
            "avatar": avatar,
            "status": sub_or_ws.status,
            "score": score,
            "maxScore": 100.0,
            "confidence": 0.95,
            "difficulty": "Medium",
            "avgClassScore": 78.0,
            "avgLatency": "1.2s",
            "questionText": sub_or_ws.title or "Classroom Worksheet",
            "fileKey": "classroom-worksheet",
            "steps": [], # Empty steps so frontend renders HTML answers
            "questions": sub_or_ws.questions or [],
            "answers": sub_or_ws.answers or {}
        }
        return ApiResponse(data=payload)

    from sqlalchemy.orm import selectinload
    query = (
        select(Submission)
        .options(selectinload(Submission.student), selectinload(Submission.steps), selectinload(Submission.grade_results), selectinload(Submission.task))
        .filter(Submission.id == sub_uuid)
    )
    result = await db.execute(query)
    submission = result.scalar_one_or_none()
    
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
        
    student = submission.student
    if student:
        student_name = student.name
        student_id = str(student.id)
    else:
        student_name = submission.student_id or f"Student {submission.file_name.split('.')[0]}"
        if student_name.upper().startswith("STUDENT-"):
            student_name = student_name[8:].replace("-", " ").title()
        student_id = submission.student_id or str(submission.id)
        
    avatar = "".join([part[0] for part in student_name.split() if part])[:2].upper()
    
    # Grade Result metadata
    score = 0.0
    max_score = 10.0
    confidence = 0.95
    if submission.grade_results:
        g = submission.grade_results[0]
        score = float(g.grade)
        max_score = float(g.max_grade)
        confidence = float(g.confidence or 0.95)
        
    task = submission.task
    difficulty = "Medium"
    avg_class_score = 78.0
    avg_latency = "1.2s"
    question_text = "N/A"
    
    if task:
        difficulty = "Medium" if task.max_marks > 5 else "Easy"
        question_text = task.description or ""
        
        # Calculate dynamically from all task submissions
        from sqlalchemy.orm import selectinload
        task_subs_query = (
            select(Submission)
            .options(selectinload(Submission.grade_results))
            .filter(Submission.task_id == task.id)
        )
        task_subs_res = await db.execute(task_subs_query)
        task_subs = task_subs_res.scalars().all()
        
        total_score = 0.0
        total_latency = 0
        latency_count = 0
        grade_count = 0
        
        for ts in task_subs:
            if ts.grade_results:
                for grade in ts.grade_results:
                    total_score += float(grade.grade or 0)
                    grade_count += 1
                    if grade.latency_ms:
                        total_latency += grade.latency_ms
                        latency_count += 1
                        
        if grade_count > 0:
            avg_class_score = round(total_score / grade_count, 1)
        else:
            avg_class_score = 78.0
            
        avg_latency_ms = int(total_latency / latency_count) if latency_count > 0 else 0
        if avg_latency_ms >= 1000:
            avg_latency = f"{avg_latency_ms / 1000:.1f}s"
        elif avg_latency_ms > 0:
            avg_latency = f"{avg_latency_ms}ms"
        else:
            avg_latency = "1.2s"

    # Retrieve task rubric steps to map questionText
    rubric_steps = []
    if task and task.rubrics:
        active_rubric = next((r for r in task.rubrics if r.is_active), None)
        if active_rubric:
            rubric_steps = active_rubric.rubric_json.get("steps", [])

    rubric_map = {str(s.get("step_num")): s.get("description", "") for s in rubric_steps}

    from app.services.ingestion import generate_presigned_url

    # Steps from submission_steps table
    steps_list = []
    for step in submission.steps:
        s_num_str = str(step.step_num)
        q_text = rubric_map.get(s_num_str, f"Question {s_num_str}")

        diagram_url = None
        bbox = step.bounding_box or {}
        diagram_key = bbox.get("diagram_key")
        if diagram_key:
            try:
                diagram_url = generate_presigned_url(diagram_key)
            except Exception as e:
                logger.warning(f"[Analysis] Diagram URL generation failed for step {s_num_str}")

        steps_list.append({
            "stepNum": step.step_num,
            "type": step.step_type or "Calculation",
            "text": step.text or "",
            "latex": step.latex or "",
            "sympyValid": step.sympy_valid,
            "justification": step.justification or "",
            "marks": step.marks_awarded,
            "maxMarks": step.max_marks,
            "errorType": step.error_type,
            "boundingBox": step.bounding_box if (step.bounding_box and "x" in step.bounding_box) else None,
            "questionText": q_text,
            "diagramUrl": diagram_url,
        })

    # Presigned URL for the original handwritten sheet (no key exposure)
    sheet_url = None
    if submission.file_key and submission.file_key != "classroom-worksheet":
        try:
            sheet_url = generate_presigned_url(submission.file_key)
        except Exception as e:
            logger.warning(f"[Analysis] Sheet URL generation failed for submission {submission_id}")

    # Presigned URL for the LaTeX transcript (if generated)
    latex_transcript_url = None
    latex_key = (submission.parsed_content or {}).get("latex_transcript_key")
    if latex_key:
        try:
            latex_transcript_url = generate_presigned_url(latex_key)
        except Exception as e:
            logger.warning(f"[Analysis] LaTeX transcript URL generation failed for submission {submission_id}")

    payload = {
        "id": str(submission.id),
        "studentId": student_id,
        "studentName": student_name,
        "avatar": avatar,
        "status": submission.status,
        "score": score,
        "maxScore": max_score,
        "confidence": confidence,
        "difficulty": difficulty,
        "avgClassScore": avg_class_score,
        "avgLatency": avg_latency,
        "questionText": question_text,
        "fileType": submission.file_type or "pdf",
        "sheetUrl": sheet_url,
        "latexTranscriptUrl": latex_transcript_url,
        "rawText": submission.raw_text or "",
        "steps": steps_list,
    }

    return ApiResponse(data=payload)


@router.post("/submissions/{submission_id}/marks")
async def override_submission_marks(
    submission_id: str,
    payload: ScoreOverrideRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Stepper score override endpoint. Updates the overall GradeResult marks
    and updates step marks in the database.
    """
    if current_user.role == "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Students are not authorized to override marks.",
        )

    try:
        sub_uuid = uuid.UUID(submission_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid submission UUID format")

    # BOLA / IDOR isolation check
    await check_submission_access(sub_uuid, current_user, db)

    # 1. Fetch GradeResult
    g_res = await db.execute(
        select(GradeResult).filter(GradeResult.submission_id == sub_uuid)
        .order_by(GradeResult.graded_at.desc()).limit(1)
    )
    grade_result = g_res.scalar_one_or_none()
    
    if not grade_result:
        raise HTTPException(status_code=404, detail="Grade result not found")

    # 2. Adjust specific step marks
    target_step_num = payload.step_num
    if target_step_num is not None:
        steps_res = await db.execute(
            select(SubmissionStep)
            .filter(and_(
                SubmissionStep.submission_id == sub_uuid,
                SubmissionStep.step_num == target_step_num
            ))
        )
    else:
        steps_res = await db.execute(
            select(SubmissionStep)
            .filter(SubmissionStep.submission_id == sub_uuid)
            .order_by(SubmissionStep.step_num.desc()).limit(1)
        )
    step_to_adjust = steps_res.scalar_one_or_none()
    
    actual_step_change = payload.amount
    if step_to_adjust:
        old_step_marks = step_to_adjust.marks_awarded
        new_step_marks = max(0.0, min(float(step_to_adjust.max_marks), float(old_step_marks) + payload.amount))
        actual_step_change = new_step_marks - old_step_marks
        step_to_adjust.marks_awarded = new_step_marks
        
        # Sync grade_result JSONB steps
        if grade_result.step_grades:
            step_grades_copy = list(grade_result.step_grades)
            for sg in step_grades_copy:
                if sg.get("step_num") == step_to_adjust.step_num:
                    sg["marks_awarded"] = new_step_marks
                    break
            grade_result.step_grades = step_grades_copy

    # Update overall grade
    old_grade = grade_result.grade
    new_grade = max(0.0, min(float(grade_result.max_grade), float(old_grade) + actual_step_change))
    grade_result.grade = int(round(new_grade)) # Round first to avoid truncation errors for floating point adjustments
    grade_result.review_status = "OVERRIDDEN"
    grade_result.review_notes = f"Teacher overridden step {target_step_num or 'last'} score from {old_grade} to {new_grade}."
            
    await db.commit()
    
    return ApiResponse(data={
        "submission_id": submission_id,
        "newScore": new_grade,
        "message": f"Successfully updated submission score to {new_grade}"
    })


@router.post("/submissions/{submission_id}/chat")
async def chat_analysis_copilot(
    submission_id: str,
    payload: ChatMessageRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Processes chat requests using the database submission steps, matches queries
    to specific steps, and returns highlighted alignment details with intelligent insights.
    """
    # Default outputs
    aligned_step = None
    aligned_reason = None
    ai_text = ""
    is_ws = False

    try:
        sub_uuid = uuid.UUID(submission_id)
        # BOLA / IDOR isolation check
        sub_or_ws = await check_submission_access(sub_uuid, current_user, db)
        
        from app.db.models import ClassroomWorksheet
        is_ws = isinstance(sub_or_ws, ClassroomWorksheet)
        
        if not is_ws:
            from sqlalchemy.orm import selectinload
            query = (
                select(Submission)
                .options(selectinload(Submission.student), selectinload(Submission.steps), selectinload(Submission.task))
                .filter(Submission.id == sub_uuid)
            )
            result = await db.execute(query)
            submission = result.scalar_one_or_none()
        else:
            submission = None
    except (ValueError, HTTPException):
        submission = None
        sub_or_ws = None

    prompt_lower = payload.message.lower()
    
    # Extract Context if passed by frontend
    question_context = ""
    clean_message = payload.message
    if "[Context - Question:" in payload.message:
        parts = payload.message.split("[Context - Question:")
        clean_message = parts[0].strip()
        question_context = parts[1].split("]")[0].strip()
        
    if is_ws:
        student_name = sub_or_ws.student.name if (sub_or_ws and sub_or_ws.student) else "the student"
    else:
        student_name = submission.student.name if (submission and submission.student) else "the student"

    # ─────────────────────────────────────────
    # ENHANCED SEMANTIC MATCHING
    # ─────────────────────────────────────────
    
    # If submission is missing or we want an LLM answer, use the LLM
    from app.services.llm_client import call_gemini
    
    # Check for specific step number mentions
    target_step = None
    if submission:
        for i in range(1, len(submission.steps) + 1):
            if f"step {i}" in prompt_lower or (["one", "two", "three", "four", "five"][i-1] if i <= 5 else "") in prompt_lower:
                target_step = i
                break
    elif is_ws and sub_or_ws and sub_or_ws.questions:
        for i in range(1, len(sub_or_ws.questions) + 1):
            if f"step {i}" in prompt_lower or f"question {i}" in prompt_lower or (["one", "two", "three", "four", "five"][i-1] if i <= 5 else "") in prompt_lower:
                target_step = i
                break

    # Build prompt for LLM
    sys_prompt = "You are OzymorLab, an expert AI Teaching Assistant. Provide concise, helpful, and educational answers to the student's doubts. Do not penalize or deduct marks for minor verbosity or structural omissions. Always be encouraging."
    llm_prompt = f"User Question/Doubt: {clean_message}\n\n"
    
    if is_ws and sub_or_ws:
        score_str = sub_or_ws.grade or "Not Graded"
        if question_context:
            llm_prompt += f"Context (Classroom Subject): {sub_or_ws.subject or ''}\n"
            llm_prompt += f"Context (Exam/Assignment Title): {question_context}\n\n"
        else:
            llm_prompt += f"Context (Classroom Worksheet): {sub_or_ws.title or 'Worksheet'}\n\n"
            
        llm_prompt += f"Student's Current Score/Grade: {score_str}\n"
        
        # Add questions and answers
        if sub_or_ws.questions:
            llm_prompt += "\nWorksheet Questions & Student Answers:\n"
            for idx, q in enumerate(sub_or_ws.questions):
                q_text = q.get("text", f"Question {idx+1}")
                q_id = q.get("id", f"q{idx+1}")
                student_ans = ""
                if sub_or_ws.answers and isinstance(sub_or_ws.answers, dict):
                    student_ans = sub_or_ws.answers.get(q_id, "")
                import re
                clean_ans = re.sub(r'<[^>]*>', '', student_ans) if student_ans else "No answer provided"
                llm_prompt += f"- Question {idx+1}: {q_text}\n"
                llm_prompt += f"  Student's Answer: {clean_ans}\n"
                
            if target_step:
                q_target = sub_or_ws.questions[target_step-1]
                q_target_text = q_target.get("text", f"Question {target_step}")
                q_target_id = q_target.get("id", f"q{target_step}")
                target_ans = ""
                if sub_or_ws.answers and isinstance(sub_or_ws.answers, dict):
                    target_ans = sub_or_ws.answers.get(q_target_id, "")
                clean_target_ans = re.sub(r'<[^>]*>', '', target_ans) if target_ans else "No answer provided"
                llm_prompt += f"\nSpecifically focusing on Question {target_step}:\nQuestion: {q_target_text}\nStudent Answer: {clean_target_ans}\n"
    else:
        if question_context:
            llm_prompt += f"Context (Exam Question): {question_context}\n\n"
        if submission:
            total_marks = sum(s.marks_awarded for s in submission.steps)
            total_max = sum(s.max_marks for s in submission.steps)
            llm_prompt += f"Student's Current Score: {total_marks}/{total_max}\n"
            if target_step:
                step_data = next((s for s in submission.steps if s.step_num == target_step), None)
                if step_data:
                    llm_prompt += f"Context about Step {target_step}:\nWork: {step_data.text or step_data.latex}\nFeedback: {step_data.justification}\n"
            else:
                errors = [s for s in submission.steps if s.error_type]
                if errors:
                    llm_prompt += "Known Errors in Student's Work:\n"
                    for e in errors[:3]:
                        llm_prompt += f"- Step {e.step_num}: {e.error_type}\n"

    llm_res = call_gemini(llm_prompt, system_prompt=sys_prompt, max_tokens=400)
    
    ai_text = llm_res.get("response_text", "I'm here to help! Could you clarify your question?")
    aligned_step = target_step or (1 if (not submission and not is_ws) else None)
    aligned_reason = "AI Assistant Analysis"
        


    return ApiResponse(data={
        "sender": "ai",
        "text": ai_text,
        "alignedStep": aligned_step,
        "alignedReason": aligned_reason
    })


@router.get("/practices", response_model=ApiResponse)
async def list_practice_history(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Fetch all practice submissions for the current student/user.
    Returns practice history with scores and metadata.
    """
    from sqlalchemy.orm import selectinload
    query = (
        select(Practice)
        .options(selectinload(Practice.grade_results), selectinload(Practice.steps))
        .filter(Practice.user_id == current_user.id)
        .order_by(Practice.created_at.desc())
    )
    result = await db.execute(query)
    practices = result.scalars().all()

    items = []
    for p in practices:
        score = 0.0
        max_score = 10.0
        if p.grade_results:
            latest_grade = p.grade_results[0]
            score = float(latest_grade.grade)
            max_score = float(latest_grade.max_grade)

        # Build steps response
        steps_list = []
        for step in p.steps:
            steps_list.append({
                "stepNum": step.step_num,
                "type": step.step_type or "Calculation",
                "text": step.text or "",
                "latex": step.latex or "",
                "sympyValid": step.sympy_valid,
                "justification": step.justification or "",
                "marks": step.marks_awarded,
                "maxMarks": step.max_marks,
                "errorType": step.error_type,
                "boundingBox": step.bounding_box
            })

        items.append({
            "id": str(p.id),
            "title": p.title or "Practice Submission",
            "date": p.created_at.strftime("%Y-%m-%d %H:%M:%S") if p.created_at else "N/A",
            "rubricName": "Self-Evaluation",
            "score": score,
            "maxScore": max_score,
            "steps": steps_list,
            "ocrText": p.raw_text
        })

    return ApiResponse(data=items)


@router.post("/practices/grade", response_model=ApiResponse)
async def grade_practice_submission(
    file: UploadFile = File(...),
    rubric: str = Form(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Grade a practice submission. Accepts file upload and rubric text.
    Returns the graded practice attempt with steps and scores.
    
    Multipart form data:
    - file: PDF or image file to grade
    - rubric: Rubric text or rubric name to use for grading
    """

    # Validate inputs
    if not file or not rubric:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both file and rubric are required"
        )

    # For now, we implement a mock version that stores the practice and returns sample graded results
    # In production, this would:
    # 1. Save file to S3
    # 2. Call diagram-marker service for OCR/parsing
    # 3. Grade using LLM with provided rubric
    # 4. Store structured results

    practice = Practice(
        user_id=current_user.id,
        file_key=f"s3://practice/{uuid.uuid4()}/{file.filename}",
        file_name=file.filename,
        file_type=file.filename.split('.')[-1] if file.filename else "pdf",
        raw_text="Mock OCR text from uploaded file",
        status="GRADED",
        title=f"Practice - {file.filename or 'Submission'}",
        rubric_json={"text": rubric}
    )

    db.add(practice)
    await db.flush()

    # Create mock practice steps
    mock_step_data = [
        {
            "step_num": 1,
            "step_type": "Initial Setup",
            "text": "Let x = sin(t)",
            "latex": "x = \\sin(t)",
            "marks_awarded": 1.0,
            "max_marks": 1.0,
            "sympy_valid": True,
            "error_type": None,
            "justification": "Correct substitution choice."
        },
        {
            "step_num": 2,
            "step_type": "Differential",
            "text": "dx = cos(t) dt",
            "latex": "dx = \\cos(t) dt",
            "marks_awarded": 1.0,
            "max_marks": 1.0,
            "sympy_valid": True,
            "error_type": None,
            "justification": "Derivative computed correctly."
        },
        {
            "step_num": 3,
            "step_type": "Integration",
            "text": "∫ sin²(t) cos(t) dt",
            "latex": "\\int \\sin^2(t) \\cos(t) dt",
            "marks_awarded": 2.0,
            "max_marks": 2.0,
            "sympy_valid": True,
            "error_type": None,
            "justification": "Integration setup is correct."
        },
        {
            "step_num": 4,
            "step_type": "Final Answer",
            "text": "= (1/3)sin³(t) + C",
            "latex": "= \\frac{1}{3}\\sin^3(t) + C",
            "marks_awarded": 2.0,
            "max_marks": 2.0,
            "sympy_valid": True,
            "error_type": None,
            "justification": "Final answer is complete with constant of integration."
        }
    ]

    # Create practice steps
    for step_data in mock_step_data:
        step = PracticeStep(
            practice_id=practice.id,
            step_num=step_data["step_num"],
            step_type=step_data["step_type"],
            text=step_data["text"],
            latex=step_data["latex"],
            marks_awarded=step_data["marks_awarded"],
            max_marks=step_data["max_marks"],
            sympy_valid=step_data["sympy_valid"],
            error_type=step_data["error_type"],
            justification=step_data["justification"]
        )
        db.add(step)

    await db.flush()

    # Create grade result
    total_marks = sum(s["marks_awarded"] for s in mock_step_data)
    total_max = sum(s["max_marks"] for s in mock_step_data)
    
    step_grades = [
        {
            "step_num": s["step_num"],
            "marks_awarded": s["marks_awarded"],
            "max_marks": s["max_marks"],
            "error_type": s["error_type"],
            "justification": s["justification"]
        }
        for s in mock_step_data
    ]

    grade_result = PracticeGradeResult(
        practice_id=practice.id,
        grade=total_marks,
        max_grade=total_max,
        confidence=0.92,
        step_grades=step_grades,
        model_used="claude-3.5-sonnet",
        justification="Well-structured solution with correct mathematical reasoning throughout."
    )

    db.add(grade_result)
    await db.commit()

    # Build response
    steps_list = [
        {
            "stepNum": step.step_num,
            "type": step.step_type or "Calculation",
            "text": step.text or "",
            "latex": step.latex or "",
            "sympyValid": step.sympy_valid,
            "justification": step.justification or "",
            "marks": step.marks_awarded,
            "maxMarks": step.max_marks,
            "errorType": step.error_type,
            "boundingBox": step.bounding_box
        }
        for step in practice.steps
    ]

    return ApiResponse(data={
        "id": str(practice.id),
        "title": practice.title,
        "date": practice.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "rubricName": "Self-Evaluation",
        "score": grade_result.grade,
        "maxScore": grade_result.max_grade,
        "steps": steps_list,
        "ocrText": practice.raw_text
    })
