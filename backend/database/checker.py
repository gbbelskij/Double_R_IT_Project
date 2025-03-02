import sqlite3


def create_database():
  connection = sqlite3.connect("backend/database/courses.db")
  cursor = connection.cursor()
  cursor.execute("SELECT id, title, direction FROM courses")
  for course in cursor.fetchall():
    print(course)
  return cursor

def main():
  create_database()

if __name__ == "__main__":
  main()