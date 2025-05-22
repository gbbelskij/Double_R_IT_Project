import csv
import uuid
from sqlalchemy import text

from backend.database.User import Course, db


def load_courses(app):
    with app.app_context():
        from flask_migrate import upgrade

        upgrade()

        with db.engine.connect() as connection:
            connection.execute(text("TRUNCATE TABLE courses CASCADE;"))
            connection.commit()

        with open("courses.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Обрабатываем поле tags вручную
                tags = [tag.strip() for tag in row["tags"].strip("[]").split(",")]
                print(tags)
                course = Course(
                    course_id=uuid.uuid4(),
                    title=row["title"],
                    link=row["link"],
                    duration=row["duration"],
                    description=row["description"],
                    price=row["price"],
                    type=row["type"],
                    direction=row["direction"],
                    tags=tags  # сохранится в JSONB
                )


                db.session.add(course)

            db.session.commit()
