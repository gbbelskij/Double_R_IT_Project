from sqlalchemy.dialects.postgresql import TEXT
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import UUID
from User import db
import uuid



class Course(db.Model):
    __tablename__ = 'courses'

    course_id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, nullable=False)
    title = db.Column(TEXT),
    link = db.Column(TEXT),
    duration = db.Column(TEXT),
    description = db.Column(TEXT),
    price = db.Column(TEXT),
    type = db.Column(TEXT),
    direction = db.Column(TEXT)
