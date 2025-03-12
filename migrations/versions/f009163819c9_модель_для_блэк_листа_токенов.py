"""Модель для блэк листа токенов

Revision ID: f009163819c9
Revises: 6593020dd566
Create Date: 2025-03-11 13:26:48.586314

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f009163819c9'
down_revision = '6593020dd566'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('tokens_black_list',
    sa.Column('token_id', sa.UUID(), nullable=False),
    sa.Column('jti', sa.String(length=36), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('token_id'),
    sa.UniqueConstraint('token_id'),
    sa.UniqueConstraint('user_id')
    )
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.create_unique_constraint(None, ['user_id'])

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='unique')

    op.drop_table('tokens_black_list')
    # ### end Alembic commands ###
