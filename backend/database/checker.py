import sqlite3

conn = sqlite3.connect("courses.db")
cursor = conn.cursor()

# Извлечение всех записей
cursor.execute("SELECT * FROM courses")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
