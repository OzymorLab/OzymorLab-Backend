"""
Reports API — Task summaries, school dashboard, and student report card PDF generation.

Provides aggregate analytics and downloadable report cards for the Edexia platform.
"""
import io
import logging
import statistics
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.db.models import (
    User, Task, Submission, GradeResult, GradingRun,
    Student, Section, SchoolClass, School, ExamCycle,
)
from app.schemas.common import ApiResponse
from app.schemas.operations import TaskSummaryResponse, SchoolDashboardResponse
from app.services.auth_service import (
    get_current_user,
    require_role,
    check_task_access,
    check_student_access
)
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/reports",
    tags=["Reports"],
    dependencies=[Depends(require_role(["teacher", "admin", "hod", "principal", "student"]))],
)


@router.get("/dashboard")
async def get_reports_dashboard(
    class_name: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get aggregate reports dashboard data including list of students,
    their classes, exams, and calculated average percentages.
    """
    # Use school fallback if missing
    school_id = current_user.school_id
    if not school_id:
        import uuid as _uuid
        school_id = _uuid.UUID("150196c5-deb3-4580-9db7-80a75de6c382")

    # Fetch students in user's school
    stmt = (
        select(Student)
        .join(Section, Student.section_id == Section.id)
        .join(SchoolClass, Section.class_id == SchoolClass.id)
        .filter(SchoolClass.school_id == school_id)
    )
    if class_name:
        stmt = stmt.filter(SchoolClass.name == class_name)
        
    res = await db.execute(stmt)
    students = res.scalars().all()
    
    student_items = []
    
    for s in students:
        # Fetch submissions and grades for this student
        from sqlalchemy.orm import selectinload
        sub_stmt = (
            select(Submission)
            .options(selectinload(Submission.grade_results))
            .filter(Submission.student_id == s.id)
        )
        sub_res = await db.execute(sub_stmt)
        submissions = sub_res.scalars().all()
        
        total_exams = len(submissions)
        total_percentage = 0.0
        graded_count = 0
        
        for sub in submissions:
            if sub.grade_results:
                latest_grade = sorted(sub.grade_results, key=lambda g: g.graded_at or datetime.min, reverse=True)[0]
                if latest_grade.max_grade > 0:
                    total_percentage += (latest_grade.grade / latest_grade.max_grade) * 100
                    graded_count += 1
                    
        avg_percentage = round(total_percentage / graded_count, 1) if graded_count > 0 else 0.0
        
        section = s.section
        school_class = section.school_class if section else None
        
        student_items.append({
            "student_id": str(s.id),
            "student_name": s.name,
            "class_name": school_class.name if school_class else "N/A",
            "section_name": section.name if section else "N/A",
            "total_exams": total_exams,
            "average_percentage": avg_percentage
        })
        
    # Build some mock/real stats summary
    stats = {
        "overall_average": round(sum(s["average_percentage"] for s in student_items) / len(student_items), 1) if student_items else 0.0,
        "total_evaluated_papers": sum(s["total_exams"] for s in student_items),
        "total_active_subjects": 3,
    }
    
    return ApiResponse(data={
        "students": student_items,
        "stats": stats
    })


@router.get("/task/{task_id}/summary")
async def get_task_summary(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get aggregate grading statistics for a specific task."""
    try:
        task_uuid = uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task UUID format")

    # BOLA / IDOR isolation check
    task = await check_task_access(task_uuid, current_user, db)

    # Fetch all grade results for this task
    grade_query = (
        select(GradeResult)
        .join(Submission, GradeResult.submission_id == Submission.id)
        .filter(Submission.task_id == task_uuid)
    )
    grade_result = await db.execute(grade_query)
    grades = grade_result.scalars().all()

    if not grades:
        return ApiResponse(data=TaskSummaryResponse(
            task_id=task_id,
            task_title=task.title,
            total_submissions=0,
            graded_count=0,
            mean_score=0.0,
            median_score=0.0,
            min_score=0.0,
            max_score=0.0,
            score_distribution=[0] * 10,
        ))

    # Count total submissions
    sub_count_result = await db.execute(
        select(func.count(Submission.id)).filter_by(task_id=task_uuid)
    )
    total_submissions = sub_count_result.scalar() or 0

    # Compute statistics
    scores = [g.grade for g in grades]
    max_possible = task.max_marks or 100

    # Normalize scores to percentages for distribution
    percentages = [(s / max_possible) * 100 if max_possible > 0 else 0 for s in scores]

    # Build histogram: 10 bins (0-10%, 10-20%, ..., 90-100%)
    distribution = [0] * 10
    for pct in percentages:
        bin_idx = min(int(pct / 10), 9)
        distribution[bin_idx] += 1

    return ApiResponse(data=TaskSummaryResponse(
        task_id=task_id,
        task_title=task.title,
        total_submissions=total_submissions,
        graded_count=len(grades),
        mean_score=round(statistics.mean(scores), 2),
        median_score=round(statistics.median(scores), 2),
        min_score=min(scores),
        max_score=max(scores),
        score_distribution=distribution,
    ))


@router.get("/school/dashboard", dependencies=[Depends(require_role(["admin", "principal"]))])
async def get_school_dashboard(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """School-wide aggregate dashboard data for principals and admins."""
    if not current_user.school_id:
        raise HTTPException(status_code=403, detail="You must belong to a school.")

    school_id = current_user.school_id

    # Fetch school name
    school_result = await db.execute(select(School).filter_by(id=school_id))
    school = school_result.scalar_one_or_none()
    if not school:
        raise HTTPException(status_code=404, detail="School not found.")

    # Aggregate counts
    teacher_count = (await db.execute(
        select(func.count(User.id)).filter_by(school_id=school_id, is_active=True)
    )).scalar() or 0

    student_count = (await db.execute(
        select(func.count(Student.id))
        .join(Section, Student.section_id == Section.id)
        .join(SchoolClass, Section.class_id == SchoolClass.id)
        .filter(SchoolClass.school_id == school_id)
    )).scalar() or 0

    cycle_count = (await db.execute(
        select(func.count(ExamCycle.id)).filter_by(school_id=school_id)
    )).scalar() or 0

    # For tasks and submissions, we scope via exam_cycles or via created_by users in this school
    task_count = (await db.execute(
        select(func.count(Task.id))
        .outerjoin(ExamCycle, Task.exam_cycle_id == ExamCycle.id)
        .filter(
            (ExamCycle.school_id == school_id) | (Task.exam_cycle_id.is_(None))
        )
    )).scalar() or 0

    submission_count = (await db.execute(
        select(func.count(Submission.id))
        .join(Task, Submission.task_id == Task.id)
        .join(ExamCycle, Task.exam_cycle_id == ExamCycle.id)
        .filter(ExamCycle.school_id == school_id)
    )).scalar() or 0

    graded_count = (await db.execute(
        select(func.count(Submission.id))
        .join(Task, Submission.task_id == Task.id)
        .join(ExamCycle, Task.exam_cycle_id == ExamCycle.id)
        .filter(ExamCycle.school_id == school_id)
        .filter(Submission.status == "GRADED")
    )).scalar() or 0

    return ApiResponse(data=SchoolDashboardResponse(
        school_id=str(school_id),
        school_name=school.name,
        total_teachers=teacher_count,
        total_students=student_count,
        total_exam_cycles=cycle_count,
        total_tasks=task_count,
        total_submissions=submission_count,
        total_graded=graded_count,
    ))


@router.get("/student/{student_id}/pdf")
async def generate_student_report_card(
    student_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        student_uuid = uuid.UUID(student_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid student UUID format")

    # BOLA / IDOR isolation check
    student = await check_student_access(student_uuid, current_user, db)

    section = student.section
    school_class = section.school_class if section else None

    # Fetch all submissions and grades for this student
    sub_result = await db.execute(
        select(Submission)
        .options(selectinload(Submission.grade_results), selectinload(Submission.task))
        .filter_by(student_id=student_uuid)
    )
    submissions = sub_result.scalars().all()

    # Build PDF using reportlab
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch, cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="reportlab is not installed. Install it with: pip install reportlab",
        )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=12,
    )
    elements.append(Paragraph("EDEXIA — Student Report Card", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    # Student Info
    info_data = [
        ["Student Name:", student.name],
        ["Roll Number:", student.roll_number],
        ["Class:", school_class.name if school_class else "N/A"],
        ["Section:", section.name if section else "N/A"],
        ["Generated:", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")],
    ]
    info_table = Table(info_data, colWidths=[2 * inch, 4 * inch])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.4 * inch))

    # Grades Table
    if submissions:
        grade_data = [["Subject", "Task", "Marks", "Max Marks", "Percentage", "Status"]]

        for sub in submissions:
            task = sub.task
            latest_grade = None
            if sub.grade_results:
                latest_grade = sorted(sub.grade_results, key=lambda g: g.graded_at or datetime.min, reverse=True)[0]

            marks = latest_grade.grade if latest_grade else "-"
            max_marks = latest_grade.max_grade if latest_grade else (task.max_marks if task else "-")
            pct = f"{(latest_grade.grade / latest_grade.max_grade * 100):.1f}%" if latest_grade and latest_grade.max_grade > 0 else "-"
            status = sub.status

            grade_data.append([
                task.subject if task else "N/A",
                task.title[:40] if task else "N/A",
                str(marks),
                str(max_marks),
                pct,
                status,
            ])

        grade_table = Table(grade_data, colWidths=[1.2 * inch, 1.8 * inch, 0.7 * inch, 0.8 * inch, 0.9 * inch, 0.8 * inch])
        grade_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (2, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(Paragraph("Academic Performance", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(grade_table)
    else:
        elements.append(Paragraph("No submissions found for this student.", styles["Normal"]))

    # Footer
    elements.append(Spacer(1, 0.5 * inch))
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    elements.append(Paragraph(
        "This report card was generated by Edexia AIOS. "
        "Grades are AI-evaluated with teacher moderation.",
        footer_style,
    ))

    doc.build(elements)
    buffer.seek(0)

    filename = f"report_card_{student.roll_number}_{student.name.replace(' ', '_')}.pdf"

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
