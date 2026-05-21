"""Add idempotency_keys table

Revision ID: 558d19a2b8e9
Revises: a4c7e1b23f90
Create Date: 2026-05-21 16:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '558d19a2b8e9'
down_revision: Union[str, None] = 'a4c7e1b23f90'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- New table: idempotency_keys ---
    op.create_table('idempotency_keys',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('key_hash', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='PROCESSING'),
        sa.Column('response_code', sa.Integer(), nullable=True),
        sa.Column('response_body', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_idempotency_keys_key_hash', 'idempotency_keys', ['key_hash'], unique=True)


def downgrade() -> None:
    # --- Drop table: idempotency_keys ---
    op.drop_index('idx_idempotency_keys_key_hash', table_name='idempotency_keys')
    op.drop_table('idempotency_keys')
