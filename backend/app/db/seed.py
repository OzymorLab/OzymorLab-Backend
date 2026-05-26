import uuid
import logging
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import (
    School,
    SchoolClass,
    Section,
    Student,
    Task,
    Submission,
    SubmissionStep,
    GradeResult,
    GradingRun,
    User
)

logger = logging.getLogger(__name__)

async def seed_database(db: AsyncSession):
    """Seed the database with high-fidelity math submission and step trace data."""
    # 1. Check if seeded already by querying for our default task
    task_check = await db.execute(select(Task).filter_by(title="Question 1: Integration by Parts"))
    if task_check.scalar_one_or_none():
        logger.info("Database is already seeded. Skipping seeder.")
        return

    logger.info("Seeding database with high-fidelity evaluation and trace data...")

    # 2. Create a default school
    school = School(name="OzymorLab Academic Academy")
    db.add(school)
    await db.flush()

    # 3. Create a class and section
    school_class = SchoolClass(school_id=school.id, name="Grade 12")
    db.add(school_class)
    await db.flush()

    section = Section(class_id=school_class.id, name="A")
    db.add(section)
    await db.flush()

    # 4. Create default mock students matching the frontend roster
    # roll_number -> name, avatar, base_score, flagColor, errorType
    students_data = [
        {"roll": "S01", "name": "Amelia Vance", "avatar": "AV", "score": 10.0, "flag": "green-d", "error": None},
        {"roll": "S02", "name": "Bhavya Patel", "avatar": "BP", "score": 9.0, "flag": "green", "error": "Minor Notation"},
        {"roll": "S03", "name": "Chris Evans", "avatar": "CE", "score": 8.0, "flag": "green-l", "error": "Incomplete Step"},
        {"roll": "S04", "name": "Daniel Craig", "avatar": "DC", "score": 0.0, "flag": "white", "error": "Unsubmitted"},
        {"roll": "S05", "name": "Emily Blunt", "avatar": "EB", "score": 6.0, "flag": "red-l", "error": "Algebraic Sign Error"},
        {"roll": "S06", "name": "Farhan Akhtar", "avatar": "FA", "score": 4.5, "flag": "red", "error": "Arithmetic Flub"},
        {"roll": "S07", "name": "Grace Hopper", "avatar": "GH", "score": 2.0, "flag": "red-d", "error": "Critical Misconception"},
    ]

    student_objects = {}
    for sd in students_data:
        student = Student(
            section_id=section.id,
            roll_number=sd["roll"],
            name=sd["name"]
        )
        db.add(student)
        student_objects[sd["roll"]] = student

    await db.flush()

    # 5. Create default tasks (questions)
    task1 = Task(
        title="Question 1: Integration by Parts",
        subject="Mathematics",
        board="CBSE",
        grade_level="Class 12",
        max_marks=10,
        description="Indefinite integration by parts question.",
        question_paper_key="q1_paper_key"
    )
    task2 = Task(
        title="Question 2: Finding Determinants",
        subject="Mathematics",
        board="CBSE",
        grade_level="Class 12",
        max_marks=10,
        description="Finding the determinant of a 2x2 matrix.",
        question_paper_key="q2_paper_key"
    )
    task3 = Task(
        title="Question 3: Quadratic Factoring",
        subject="Mathematics",
        board="CBSE",
        grade_level="Class 12",
        max_marks=10,
        description="Solving quadratic equation by factoring.",
        question_paper_key="q3_paper_key"
    )

    db.add_all([task1, task2, task3])
    await db.flush()

    # Create a dummy grading run to satisfy constraints
    run = GradingRun(
        task_id=task1.id,
        rubric_version="1.0.0",
        model="gemini-2.5-pro",
        temperature=0.0,
        description="Seeded Grading Run",
        status="COMPLETED",
        total_submissions=len(students_data),
        graded_count=len(students_data),
        failed_count=0
    )
    db.add(run)
    await db.flush()

    # 6. Populate submissions, steps, and grades for Q1
    # S01 Integration Steps (100% correct)
    steps_s01_q1 = [
        {"step_num": 1, "type": "Substitution Setup", "text": "Assign u and dv terms for integration by parts.", "latex": "u = x, \\quad dv = e^{2x} \\, dx", "valid": True, "just": "Valid variables selected. u is simpler to differentiate.", "marks": 3.0, "max": 3.0, "err": None, "box": {"x": 8, "y": 15, "w": 84, "h": 12}},
        {"step_num": 2, "type": "Integration", "text": "Differentiate u and integrate dv to get du and v.", "latex": "du = dx, \\quad v = \\frac{1}{2}e^{2x}", "valid": True, "just": "Calculated derivatives and integrals accurately using standard rules.", "marks": 3.0, "max": 3.0, "err": None, "box": {"x": 8, "y": 32, "w": 84, "h": 14}},
        {"step_num": 3, "type": "IBP Formula Expansion", "text": "Apply the integration by parts formula: \\int u \\, dv = uv - \\int v \\, du.", "latex": "\\int x e^{2x} \\, dx = \\frac{1}{2}x e^{2x} - \\int \\frac{1}{2}e^{2x} \\, dx", "valid": True, "just": "Perfect substitution into formula structure.", "marks": 2.0, "max": 2.0, "err": None, "box": {"x": 8, "y": 50, "w": 84, "h": 16}},
        {"step_num": 4, "type": "Final Solution", "text": "Evaluate the final integral and add the integration constant C.", "latex": "= \\frac{1}{2}x e^{2x} - \\frac{1}{4}e^{2x} + C", "valid": True, "just": "Integration completed successfully with constant included.", "marks": 2.0, "max": 2.0, "err": None, "box": {"x": 8, "y": 72, "w": 84, "h": 15}}
    ]

    # S02 Integration Steps (notation error - missing + C)
    steps_s02_q1 = [
        {"step_num": 1, "type": "Substitution Setup", "text": "Assign u and dv terms for integration by parts.", "latex": "u = x, \\quad dv = e^{2x} \\, dx", "valid": True, "just": "Correct assignment.", "marks": 3.0, "max": 3.0, "err": None, "box": {"x": 8, "y": 15, "w": 84, "h": 12}},
        {"step_num": 2, "type": "Integration", "text": "Differentiate u and integrate dv.", "latex": "du = dx, \\quad v = \\frac{1}{2}e^{2x}", "valid": True, "just": "Calculated correctly.", "marks": 3.0, "max": 3.0, "err": None, "box": {"x": 8, "y": 32, "w": 84, "h": 14}},
        {"step_num": 3, "type": "IBP Formula Expansion", "text": "Substitute into parts formula.", "latex": "\\int x e^{2x} \\, dx = \\frac{1}{2}x e^{2x} - \\int \\frac{1}{2}e^{2x} \\, dx", "valid": True, "just": "Accurate parts layout.", "marks": 2.0, "max": 2.0, "err": None, "box": {"x": 8, "y": 50, "w": 84, "h": 16}},
        {"step_num": 4, "type": "Final Solution", "text": "Integration but missing C constant.", "latex": "= \\frac{1}{2}x e^{2x} - \\frac{1}{4}e^{2x}", "valid": None, "just": "Calculated arithmetic, but missing integration constant C.", "marks": 1.0, "max": 2.0, "err": "Minor Notation", "box": {"x": 8, "y": 72, "w": 84, "h": 15}}
    ]

    # S05 Integration Steps (coefficient error)
    steps_s05_q1 = [
        {"step_num": 1, "type": "Substitution Setup", "text": "Assign u and dv terms.", "latex": "u = x, \\quad dv = e^{2x} \\, dx", "valid": True, "just": "Correct initial choice.", "marks": 3.0, "max": 3.0, "err": None, "box": {"x": 8, "y": 15, "w": 84, "h": 12}},
        {"step_num": 2, "type": "Integration", "text": "Incorrectly integrated dv (missing 1/2 factor).", "latex": "du = dx, \\quad v = e^{2x}", "valid": False, "just": "Integration step failed: dv = e^{2x} integrates to v = 1/2 e^{2x}. Missing coefficient 1/2.", "marks": 1.0, "max": 3.0, "err": "Algebraic Coefficient Error", "box": {"x": 8, "y": 32, "w": 84, "h": 14}},
        {"step_num": 3, "type": "IBP Formula Expansion", "text": "Parts formula expansion utilizing wrong v.", "latex": "\\int x e^{2x} \\, dx = x e^{2x} - \\int e^{2x} \\, dx", "valid": True, "just": "Follows logically from Step 2, but has algebraic error propagation.", "marks": 1.0, "max": 2.0, "err": "Error Propagation", "box": {"x": 8, "y": 50, "w": 84, "h": 16}},
        {"step_num": 4, "type": "Final Solution", "text": "Evaluate based on previous steps.", "latex": "= x e^{2x} - e^{2x} + C", "valid": True, "just": "Correct evaluation of erroneous integral.", "marks": 1.0, "max": 2.0, "err": "Error Propagation", "box": {"x": 8, "y": 72, "w": 84, "h": 15}}
    ]

    # S06 Integration Steps (arithmetic flub)
    steps_s06_q1 = [
        {"step_num": 1, "type": "Substitution Setup", "text": "Assign parts variables.", "latex": "u = x, \\quad dv = e^{2x} \\, dx", "valid": True, "just": "Correct setup.", "marks": 3.0, "max": 3.0, "err": None, "box": {"x": 8, "y": 15, "w": 84, "h": 12}},
        {"step_num": 2, "type": "Integration", "text": "Differentiate u and integrate dv.", "latex": "du = dx, \\quad v = 2e^{2x}", "valid": False, "just": "Calculated integration incorrectly: integrated dv as derivative 2 e^{2x} instead of 1/2 e^{2x}.", "marks": 0.5, "max": 3.0, "err": "Arithmetic Flub", "box": {"x": 8, "y": 32, "w": 84, "h": 14}},
        {"step_num": 3, "type": "IBP Formula Expansion", "text": "Expanded structure using wrong v.", "latex": "\\int x e^{2x} = 2x e^{2x} - 2 \\int e^{2x} \\, dx", "valid": True, "just": "Consistent structure but mathematically wrong coefficients.", "marks": 0.5, "max": 2.0, "err": "Error Propagation", "box": {"x": 8, "y": 50, "w": 84, "h": 16}},
        {"step_num": 4, "type": "Final Solution", "text": "Completed integral based on previous values.", "latex": "= 2x e^{2x} - e^{2x} + C", "valid": False, "just": "Integrals evaluated incorrectly: integrated 2e^{2x} as e^{2x}.", "marks": 0.5, "max": 2.0, "err": "Arithmetic Flub", "box": {"x": 8, "y": 72, "w": 84, "h": 15}}
    ]

    # S07 Integration Steps (strategic deadend)
    steps_s07_q1 = [
        {"step_num": 1, "type": "Substitution Setup", "text": "Assign parts variables backward.", "latex": "u = e^{2x}, \\quad dv = x \\, dx", "valid": True, "just": "Valid assignment but makes integration by parts harder to solve.", "marks": 2.0, "max": 3.0, "err": "Suboptimal Strategy", "box": {"x": 8, "y": 15, "w": 84, "h": 12}},
        {"step_num": 2, "type": "Integration", "text": "Evaluated parts backward.", "latex": "du = 2e^{2x} \\, dx, \\quad v = \\frac{1}{2}x^2", "valid": True, "just": "Evaluated backward variables correctly.", "marks": 0.0, "max": 3.0, "err": None, "box": {"x": 8, "y": 32, "w": 84, "h": 14}},
        {"step_num": 3, "type": "IBP Formula Expansion", "text": "Substituted into formulas producing a harder integral.", "latex": "\\int x e^{2x} \\, dx = \\frac{1}{2}x^2 e^{2x} - \\int x^2 e^{2x} \\, dx", "valid": True, "just": "Formula structured correctly, but student hit a dead end and stopped.", "marks": 0.0, "max": 2.0, "err": "Strategic Deadend", "box": {"x": 8, "y": 50, "w": 84, "h": 16}}
    ]

    submissions_q1 = [
        ("S01", steps_s01_q1, 10.0),
        ("S02", steps_s02_q1, 9.0),
        ("S05", steps_s05_q1, 6.0),
        ("S06", steps_s06_q1, 4.5),
        ("S07", steps_s07_q1, 2.0)
    ]

    for student_roll, steps_list, total_score in submissions_q1:
        st = student_objects[student_roll]
        sub = Submission(
            task_id=task1.id,
            student_id=st.id,
            file_key=f"seeded_q1_{student_roll}.png",
            file_name=f"math_submission_{student_roll}.png",
            file_type="png",
            status="GRADED"
        )
        db.add(sub)
        await db.flush()

        # Add steps
        for s in steps_list:
            step = SubmissionStep(
                submission_id=sub.id,
                step_num=s["step_num"],
                step_type=s["type"],
                text=s["text"],
                latex=s["latex"],
                sympy_valid=s["valid"],
                justification=s["just"],
                marks_awarded=s["marks"],
                max_marks=s["max"],
                error_type=s["err"],
                bounding_box=s["box"]
            )
            db.add(step)

        # Add grade result
        grade_dist = [0.0] * 11
        grade_dist[int(total_score)] = 1.0 # simplistic distribution

        step_grades_serialized = []
        for s in steps_list:
            step_grades_serialized.append({
                "step_num": s["step_num"],
                "marks_awarded": int(s["marks"]),
                "max_marks": int(s["max"]),
                "justification": s["just"],
                "error_type": s["err"],
                "sympy_valid": s["valid"],
                "grade_distribution": [0.0] * (int(s["max"]) + 1)
            })

        grade = GradeResult(
            submission_id=sub.id,
            grading_run_id=run.id,
            grade=int(total_score),
            max_grade=10,
            grade_distribution=grade_dist,
            confidence=0.96,
            step_grades=step_grades_serialized,
            justification=f"Completed grading for {st.name} with score {total_score}/10.",
            model_used="gemini-2.5-pro",
            latency_ms=1200
        )
        db.add(grade)

    # 7. Seed Q2
    # S01 correct
    steps_s01_q2 = [
        {"step_num": 1, "type": "Formula Setup", "text": "Define determinant formula for a 2x2 matrix.", "latex": "\\det(A) = ad - bc", "valid": True, "just": "Standard formula selected.", "marks": 4.0, "max": 4.0, "err": None, "box": {"x": 10, "y": 18, "w": 80, "h": 14}},
        {"step_num": 2, "type": "Substitution", "text": "Substitute coefficients: a=3, b=-2, c=5, d=4.", "latex": "= (3)(4) - (-2)(5)", "valid": True, "just": "Substituted variables perfectly with signs preserved.", "marks": 3.0, "max": 3.0, "err": None, "box": {"x": 10, "y": 38, "w": 80, "h": 15}},
        {"step_num": 3, "type": "Final Solution", "text": "Compute the determinant value.", "latex": "= 12 - (-10) = 22", "valid": True, "just": "Correctly resolved negatives and addition. Determinant is 22.", "marks": 3.0, "max": 3.0, "err": None, "box": {"x": 10, "y": 60, "w": 80, "h": 16}}
    ]

    # S06 sign loss error
    steps_s06_q2 = [
        {"step_num": 1, "type": "Formula Setup", "text": "Define formula.", "latex": "\\det(A) = ad - bc", "valid": True, "just": "Correct formula.", "marks": 4.0, "max": 4.0, "err": None, "box": {"x": 10, "y": 18, "w": 80, "h": 14}},
        {"step_num": 2, "type": "Substitution", "text": "Substituted but lost a negative sign.", "latex": "= (3)(4) - (2)(5)", "valid": False, "just": "Arithmetic substitution error: wrote (2) instead of (-2) for b term, losing a negative sign.", "marks": 0.5, "max": 3.0, "err": "Sign Signage Loss", "box": {"x": 10, "y": 38, "w": 80, "h": 15}},
        {"step_num": 3, "type": "Final Solution", "text": "Computed wrong determinant.", "latex": "= 12 - 10 = 2", "valid": True, "just": "Calculated 2 logically following Step 2's sign error.", "marks": 0.0, "max": 3.0, "err": "Error Propagation", "box": {"x": 10, "y": 60, "w": 80, "h": 16}}
    ]

    submissions_q2 = [
        ("S01", steps_s01_q2, 10.0),
        ("S06", steps_s06_q2, 4.5)
    ]

    for student_roll, steps_list, total_score in submissions_q2:
        st = student_objects[student_roll]
        sub = Submission(
            task_id=task2.id,
            student_id=st.id,
            file_key=f"seeded_q2_{student_roll}.png",
            file_name=f"math_submission_q2_{student_roll}.png",
            file_type="png",
            status="GRADED"
        )
        db.add(sub)
        await db.flush()

        for s in steps_list:
            step = SubmissionStep(
                submission_id=sub.id,
                step_num=s["step_num"],
                step_type=s["type"],
                text=s["text"],
                latex=s["latex"],
                sympy_valid=s["valid"],
                justification=s["just"],
                marks_awarded=s["marks"],
                max_marks=s["max"],
                error_type=s["err"],
                bounding_box=s["box"]
            )
            db.add(step)

        # Add grade result
        step_grades_serialized = []
        for s in steps_list:
            step_grades_serialized.append({
                "step_num": s["step_num"],
                "marks_awarded": int(s["marks"]),
                "max_marks": int(s["max"]),
                "justification": s["just"],
                "error_type": s["err"],
                "sympy_valid": s["valid"],
                "grade_distribution": [0.0] * (int(s["max"]) + 1)
            })

        grade = GradeResult(
            submission_id=sub.id,
            grading_run_id=run.id,
            grade=int(total_score),
            max_grade=10,
            grade_distribution=[0.0]*11,
            confidence=0.99,
            step_grades=step_grades_serialized,
            justification=f"Matrix determinant check graded.",
            model_used="gemini-2.5-pro",
            latency_ms=800
        )
        db.add(grade)

    # 8. Commit
    await db.commit()
    logger.info("Successfully seeded all submissions, steps, and grades tables!")
