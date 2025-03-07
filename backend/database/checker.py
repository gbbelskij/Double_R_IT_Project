import sqlite3
from backend.database.course_tags import populate_tags, populate_course_tags

# Ф-ия для просмотра содержимого базы данных
def check_database():
  connection = sqlite3.connect("backend/database/users.db")
  cursor = connection.cursor()
  cursor.execute("SELECT id, name FROM users")
  for course in cursor.fetchall():
    print(course)
  return cursor

def add_course_to_db(title, link, duration, description, price, course_type, direction, tags, course_tags):
  connection = sqlite3.connect("backend/database/courses.db")
  cursor = connection.cursor()
  cursor.execute("""
  INSERT INTO courses (title, link, duration, description, price, type, direction)
  VALUES (?, ?, ?, ?, ?, ?, ?)
  """, (title, link, duration, description, price, course_type, direction))

  populate_tags(cursor, tags)
  populate_course_tags(cursor, course_tags)

def main():
  check_database()

if __name__ == "__main__":
  main()