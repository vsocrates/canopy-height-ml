"""add scl_valid_obs to gedi_shots_cleaned

Revision ID: a1b2c3d4e5f6
Revises: 4de39ca31067
Create Date: 2026-04-22
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "4de39ca31067"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "gedi_shots_cleaned",
        sa.Column("scl_valid_obs", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("gedi_shots_cleaned", "scl_valid_obs")
