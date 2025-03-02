import pandas as pd
import numpy as np
import sqlite3

# Подключение к базе данных
conn = sqlite3.connect('backend/database/courses.db')
cursor = conn.cursor()

# Запрос для получения данных о курсах и их тегах
query = '''
    SELECT c.id, c.title, t.name
    FROM courses c
    LEFT JOIN course_tags ct ON c.id = ct.course_id
    LEFT JOIN tags t ON ct.tag_id = t.id
'''
cursor.execute(query)
rows = cursor.fetchall()

# Преобразование данных
courses = []
current_course = None

for row in rows:
    course_id, title, tag_name = row
    
    # Если это новый курс
    if not current_course or current_course["id"] != course_id:
        if current_course:
            courses.append(current_course)
        current_course = {
            "id": course_id,
            "name": title,
            "tags": []
        }
    
    # Добавляем тег, если он есть
    if tag_name:
        current_course["tags"].append(tag_name)

# Добавляем последний курс
if current_course:
    courses.append(current_course)

# Закрываем соединение
conn.close()

# Вывод результата
print(courses)

# Пользователи и их интересы
users = {
    "U1": {
        "name": "Алексей Иванов",
        "experience": 2,
        "position": "Frontend-разработчик",
        "tags": ["ui-ux", "web-design", "frontend"],
    },
    "U2": {
        "name": "Мария Петрова",
        "experience": 3,
        "position": "Backend-разработчик",
        "tags": ["backend", "python", "api"],
    },
    "U3": {
        "name": "Дмитрий Смирнов",
        "experience": 4,
        "position": "Инженер баз данных",
        "tags": ["sql", "database-design"],
    },
    "U4": {
        "name": "Елена Кузнецова",
        "experience": 5,
        "position": "Data Scientist",
        "tags": ["data-science", "machine-learning"],
    },
    "U5": {
        "name": "Игорь Васильев",
        "experience": 3,
        "position": "DevOps-инженер",
        "tags": ["automation", "scripting"],
    },
    "U6": {
        "name": "Ольга Новикова",
        "experience": 2,
        "position": "Геймдизайнер",
        "tags": ["game-development", "3d-modeling"],
    },
    "U7": {
        "name": "Анна Морозова",
        "experience": 1,
        "position": "Графический дизайнер",
        "tags": ["graphic-design", "adobe"],
    },
    "U8": {
        "name": "Сергей Волков",
        "experience": 3,
        "position": "Android-разработчик",
        "tags": ["mobile-development", "android"],
    },
    "U9": {
        "name": "Павел Белов",
        "experience": 6,
        "position": "Специалист по кибербезопасности",
        "tags": ["cybersecurity", "ethical-hacking"],
    },
    "U10": {
        "name": "Татьяна Козлова",
        "experience": 4,
        "position": "Бизнес-аналитик",
        "tags": ["data-analysis", "business-intelligence"],
    }
}

# Генерация матрицы
matrix = []
for course in courses:
    row = {}
    for user_id, user_data in users.items():
        # Расчет балла
        common_tags = set(course["tags"]) & set(user_data["tags"])
        score = 0.0
        if common_tags:
            score = 0.6 + 0.1 * len(common_tags)
            # Учет уровня сложности (пример: beginner → +0.2)
            if "beginner" in course["tags"] and user_data["experience"] < 1:
                score += 0.2
            if "advanced" in course["tags"] and user_data["experience"] > 4:
                score += 0.2
            if "intermediate" in course["tags"] and (1 <= user_data["experience"] <= 4):
                score += 0.2
        # Добавление шума
        score += np.random.uniform(-0.05, 0.05)
        row[user_id] = round(np.clip(score, 0, 1), 2)
    matrix.append(row)

# Преобразование в DataFrame
df = pd.DataFrame(matrix)
df.to_csv("course_user_matrix.csv", index=False)