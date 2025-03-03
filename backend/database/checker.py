import sqlite3
from tags import populate_tags, populate_course_tags


def create_database():
  connection = sqlite3.connect("backend/database/courses.db")
  cursor = connection.cursor()
  cursor.execute("SELECT id, title, link FROM courses")
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
  create_database()

if __name__ == "__main__":
  main()