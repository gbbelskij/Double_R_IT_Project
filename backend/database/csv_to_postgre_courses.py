from backend.database.User import Course, db
import csv



def load_csv_to_courses_db(csv_file_path):
    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # Читаем CSV как словарь

        for row in reader:
            course = Course(
                course_id=row['course_id'],  # Генерируем UUID
                title=row["title"],
                link=row["link"],
                duration=row["duration"],
                description=row["description"],
                price=row["price"],
                type=row["type"],
                direction=row["direction"]
            )
            db.session.add(course)

        db.session.commit()
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
