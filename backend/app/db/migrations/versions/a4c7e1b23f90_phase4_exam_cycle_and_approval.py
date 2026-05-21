"""Phase 4: Exam cycle model, task paper_set, rubric approval workflow

Revision ID: a4c7e1b23f90
Revises: 9b88d3fc71e7
Create Date: 2026-05-21 15:25:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a4c7e1b23f90'
down_revision: Union[str, None] = '9b88d3fc71e7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- New table: exam_cycles ---
    op.create_table('exam_cycles',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('school_id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='ACTIVE'),
        sa.Column('created_by', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['school_id'], ['schools.id']),
        sa.ForeignKeyConstraint(['created_by'], ['users.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_exam_cycles_school_id', 'exam_cycles', ['school_id'], unique=False)

    # --- tasks: add exam_cycle_id and paper_set (nullable for backward compat) ---
    op.add_column('tasks', sa.Column('exam_cycle_id', sa.UUID(), nullable=True))
    op.add_column('tasks', sa.Column('paper_set', sa.String(length=10), nullable=True))
    op.create_index(op.f('ix_tasks_exam_cycle_id'), 'tasks', ['exam_cycle_id'], unique=False)
    op.create_foreign_key('fk_tasks_exam_cycle_id', 'tasks', 'exam_cycles', ['exam_cycle_id'], ['id'])

    # --- task_rubrics: add approval workflow columns ---
    op.add_column('task_rubrics', sa.Column('approval_status', sa.String(length=20), nullable=False, server_default='DRAFT'))
    op.add_column('task_rubrics', sa.Column('approved_by', sa.UUID(), nullable=True))
    op.add_column('task_rubrics', sa.Column('approved_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('task_rubrics', sa.Column('rejection_notes', sa.Text(), nullable=True))
    op.create_index('idx_task_rubrics_approval_status', 'task_rubrics', ['approval_status'], unique=False)
    op.create_foreign_key('fk_task_rubrics_approved_by', 'task_rubrics', 'users', ['approved_by'], ['id'])


def downgrade() -> None:
    # --- task_rubrics: remove approval columns ---
    op.drop_constraint('fk_task_rubrics_approved_by', 'task_rubrics', type_='foreignkey')
    op.drop_index('idx_task_rubrics_approval_status', table_name='task_rubrics')
    op.drop_column('task_rubrics', 'rejection_notes')
    op.drop_column('task_rubrics', 'approved_at')
    op.drop_column('task_rubrics', 'approved_by')
    op.drop_column('task_rubrics', 'approval_status')

    # --- tasks: remove exam_cycle columns ---
    op.drop_constraint('fk_tasks_exam_cycle_id', 'tasks', type_='foreignkey')
    op.drop_index(op.f('ix_tasks_exam_cycle_id'), table_name='tasks')
    op.drop_column('tasks', 'paper_set')
    op.drop_column('tasks', 'exam_cycle_id')

    # --- Drop exam_cycles table ---
    op.drop_index('idx_exam_cycles_school_id', table_name='exam_cycles')
    op.drop_table('exam_cycles')
