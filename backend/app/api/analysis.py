"""
Analysis HUD API — Unified queries, rosters, and contextual chat alignment.
"""
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
    boundingBox: Optional[BoundingBox]

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
        if not student_obj:
            continue
            
        student_id = str(student_obj.id)
        name = student_obj.name
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
    await check_submission_access(sub_uuid, current_user, db)

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
    student_name = student.name if student else "Unknown"
    student_id = str(student.id) if student else "N/A"
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
        # Hardcode average for telemetry mock metrics
        if "Question 1" in task.title:
            avg_class_score = 78.0
            avg_latency = "1.2s"
        elif "Question 2" in task.title:
            avg_class_score = 84.0
            avg_latency = "0.8s"
        else:
            avg_class_score = 92.0
            avg_latency = "0.6s"

    # Steps from submission_steps table
    steps_list = []
    for step in submission.steps:
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
        "fileKey": submission.file_key,
        "steps": steps_list
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

    # Update overall grade
    old_grade = grade_result.grade
    new_grade = max(0.0, min(float(grade_result.max_grade), float(old_grade) + payload.amount))
    grade_result.grade = int(new_grade) # Cast to int for schema consistency
    grade_result.review_status = "OVERRIDDEN"
    grade_result.review_notes = f"Teacher overridden score from {old_grade} to {new_grade} using interactive marks stepper."
    
    # 2. Proportionally adjust the last step's marks in both table and JSONB to keep them in sync
    steps_res = await db.execute(
        select(SubmissionStep)
        .filter(SubmissionStep.submission_id == sub_uuid)
        .order_by(SubmissionStep.step_num.desc()).limit(1)
    )
    last_step = steps_res.scalar_one_or_none()
    if last_step:
        last_step.marks_awarded = max(0.0, min(float(last_step.max_marks), float(last_step.marks_awarded) + payload.amount))

    # Also sync grade_result JSONB steps
    if grade_result.step_grades:
        step_grades_copy = list(grade_result.step_grades)
        if len(step_grades_copy) > 0:
            step_grades_copy[-1]["marks_awarded"] = max(0.0, step_grades_copy[-1]["marks_awarded"] + payload.amount)
            grade_result.step_grades = step_grades_copy
            
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
    try:
        sub_uuid = uuid.UUID(submission_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid submission UUID format")

    # BOLA / IDOR isolation check
    await check_submission_access(sub_uuid, current_user, db)

    from sqlalchemy.orm import selectinload
    query = (
        select(Submission)
        .options(selectinload(Submission.student), selectinload(Submission.steps), selectinload(Submission.task))
        .filter(Submission.id == sub_uuid)
    )
    result = await db.execute(query)
    submission = result.scalar_one_or_none()
    
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
        
    student_name = submission.student.name if submission.student else "the student"
    prompt_lower = payload.message.lower()
    
    # Default outputs
    aligned_step = None
    aligned_reason = None
    ai_text = ""

    # ─────────────────────────────────────────
    # ENHANCED SEMANTIC MATCHING
    # ─────────────────────────────────────────
    
    # Keywords for error types
    error_keywords = {
        "sign": ["sign", "negative", "positive", "+", "-"],
        "arithmetic": ["arithmetic", "calculation", "computed", "wrong answer", "result"],
        "notation": ["notation", "constant", "missing", "format"],
        "strategy": ["strategy", "approach", "method", "technique"],
        "algebraic": ["algebra", "algebraic", "expand", "simplify"],
    }
    
    # Keywords for specific questions
    question_keywords = {
        "error": ["error", "mistake", "wrong", "incorrect", "issue", "problem"],
        "validity": ["valid", "check", "sympy", "computational", "algebra"],
        "marks": ["marks", "score", "points", "grade", "deduction"],
        "step": ["step", "part", "section", "line"],
        "integration": ["integrat", "integral"],
        "derivat": ["deriv", "derivative", "differential"],
        "substitut": ["substit", "substitute"],
        "constant": ["constant", "c", "+ c", "integration constant"],
    }
    
    # Check for specific step number mentions
    target_step = None
    for i in range(1, len(submission.steps) + 1):
        if f"step {i}" in prompt_lower or (["one", "two", "three", "four", "five"][i-1] if i <= 5 else "") in prompt_lower:
            target_step = i
            break
    
    # Find matching steps if no specific step mentioned
    if target_step:
        matched_steps = [s for s in submission.steps if s.step_num == target_step]
    else:
        # Search for steps with mentioned error types
        matched_steps = []
        for step in submission.steps:
            for error_type, keywords in error_keywords.items():
                if any(kw in prompt_lower for kw in keywords):
                    if step.error_type and error_type.lower() in step.error_type.lower():
                        matched_steps.append(step)
            # Also match by sympy validity if asking about validity
            if "valid" in prompt_lower or "check" in prompt_lower:
                if step.sympy_valid is not None:
                    matched_steps.append(step)
        
        # If still no matches, find first error
        if not matched_steps:
            invalid_step = next((s for s in submission.steps if s.sympy_valid is False or s.error_type), None)
            if invalid_step:
                matched_steps = [invalid_step]
            else:
                matched_steps = [submission.steps[0]] if submission.steps else []

    # Process matched step(s)
    if matched_steps:
        matched_step = matched_steps[0]  # Focus on primary match
        aligned_step = matched_step.step_num
        aligned_reason = matched_step.step_type or "Step analysis"
        
        # Build contextual response based on step state
        if "error" in prompt_lower or "wrong" in prompt_lower or "mistake" in prompt_lower:
            if matched_step.error_type:
                ai_text = f"I found the issue in Step {matched_step.step_num}. The error type is: **{matched_step.error_type}**.\n\n"
                ai_text += f"**Current work:** {matched_step.text or matched_step.latex}\n\n"
                ai_text += f"**Analysis:** {matched_step.justification or 'This step contains a logical error.'}\n\n"
                ai_text += f"**Suggestion:** Review the {matched_step.step_type or 'calculation'} and check your working. The highlighted section shows exactly where the issue occurs."
            else:
                ai_text = f"Step {matched_step.step_num} appears to be correct. No errors detected. SymPy validation: {'✓ PASSED' if matched_step.sympy_valid else '✗ NEEDS REVIEW' if matched_step.sympy_valid is False else '⚠ PARTIAL'}"
                
        elif "marks" in prompt_lower or "score" in prompt_lower or "why" in prompt_lower:
            awarded = matched_step.marks_awarded
            max_marks = matched_step.max_marks
            percentage = (awarded / max_marks * 100) if max_marks > 0 else 0
            ai_text = f"**Marks for Step {matched_step.step_num}:** {awarded}/{max_marks} ({percentage:.0f}%)\n\n"
            ai_text += f"**Type:** {matched_step.step_type or 'Calculation'}\n"
            ai_text += f"**Status:** {'✓ Full credit' if percentage == 100 else '✓ Partial credit' if percentage > 0 else '✗ No credit'}\n\n"
            if matched_step.error_type:
                ai_text += f"**Reason:** {matched_step.error_type}\n"
            if matched_step.justification:
                ai_text += f"**Feedback:** {matched_step.justification}"
                
        elif "valid" in prompt_lower or "check" in prompt_lower:
            ai_text = f"**SymPy Validation for Step {matched_step.step_num}:**\n\n"
            if matched_step.sympy_valid is True:
                ai_text += f"✓ **VALID** - The algebraic expression is mathematically correct.\n"
            elif matched_step.sympy_valid is False:
                ai_text += f"✗ **INVALID** - The algebraic expression contains errors.\n"
                if matched_step.error_type:
                    ai_text += f"Error: {matched_step.error_type}\n"
            else:
                ai_text += f"⚠ **UNVERIFIED** - Could not be automatically validated.\n"
            
            ai_text += f"\n**LaTeX:** `{matched_step.latex or 'N/A'}`\n"
            ai_text += f"**OCR Text:** {matched_step.text or 'N/A'}"
            
        else:
            # Generic step explanation
            ai_text = f"**Step {matched_step.step_num} Analysis:** {matched_step.step_type or 'Calculation'}\n\n"
            ai_text += f"**Work shown:** {matched_step.text or matched_step.latex or 'See highlighted section'}\n\n"
            ai_text += f"**Assessment:** {matched_step.justification or 'Step reviewed'}\n"
            ai_text += f"**Score:** {matched_step.marks_awarded}/{matched_step.max_marks}\n\n"
            if matched_step.error_type:
                ai_text += f"**⚠ Issue:** {matched_step.error_type}"
            else:
                ai_text += f"**✓ Status:** Correct"
    
    else:
        # No specific match - provide general feedback
        aligned_step = 1
        aligned_reason = "Overall assessment"
        
        total_marks = sum(s.marks_awarded for s in submission.steps)
        total_max = sum(s.max_marks for s in submission.steps)
        errors_found = [s for s in submission.steps if s.error_type]
        
        ai_text = f"**Overall Submission Analysis for {student_name}:**\n\n"
        ai_text += f"**Final Score:** {total_marks}/{total_max} ({(total_marks/total_max*100):.0f}%)\n"
        
        if errors_found:
            ai_text += f"**Issues Found:** {len(errors_found)} steps have identified errors\n\n"
            ai_text += f"**Problem Areas:**\n"
            for step in errors_found[:3]:  # Show top 3 errors
                ai_text += f"- Step {step.step_num} ({step.step_type}): {step.error_type}\n"
            if len(errors_found) > 3:
                ai_text += f"- ...and {len(errors_found) - 3} more issues\n"
        else:
            ai_text += f"**✓ Strengths:** No critical errors detected across {len(submission.steps)} steps\n"
        
        ai_text += f"\n**Try asking about:** A specific step number, error types, marks breakdown, or validation status."

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
