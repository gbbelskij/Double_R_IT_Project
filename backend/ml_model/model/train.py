# train.py
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import numpy as np
from model import CourseRecommender
from save_load import save_model
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

# Пользователи и их интересы
users = {
    "U1": {
        "name": "Алексей Иванов",
        "experience": 2,
        "position": "Frontend-разработчик",
        "tags": ["Design & UX", "Frontend Developer", "Программирование на JavaScript"],
    },
    "U2": {
        "name": "Мария Петрова",
        "experience": 3,
        "position": "Backend-разработчик",
        "tags": ["Development", "Backend Developer", "Программирование на Python"],
    },
    "U3": {
        "name": "Дмитрий Смирнов",
        "experience": 4,
        "position": "Инженер баз данных",
        "tags": ["Data & Analytics", "Data Scientist", "SQL"],
    },
    "U4": {
        "name": "Елена Кузнецова",
        "experience": 5,
        "position": "Data Scientist",
        "tags": ["Data & Analytics", "Data Scientist", "Машинное обучение", "Нейронные сети"],
    },
    "U5": {
        "name": "Игорь Васильев",
        "experience": 3,
        "position": "DevOps-инженер",
        "tags": ["Development", "DevOps Engineer", "CI/CD", "Docker", "Kubernetes"],
    },
    "U6": {
        "name": "Ольга Новикова",
        "experience": 2,
        "position": "Геймдизайнер",
        "tags": ["Design & UX", "Game Developer", "Unity", "3D-дизайн"],
    },
    "U7": {
        "name": "Анна Морозова",
        "experience": 1,
        "position": "Графический дизайнер",
        "tags": ["Design & UX", "Graphic Designer", "Adobe Photoshop", "Adobe Illustrator"],
    },
    "U8": {
        "name": "Сергей Волков",
        "experience": 3,
        "position": "Android-разработчик",
        "tags": ["Development", "Mobile Developer", "Программирование на Java", "Android"],
    },
    "U9": {
        "name": "Павел Белов",
        "experience": 6,
        "position": "Специалист по кибербезопасности",
        "tags": ["Cybersecurity", "Cybersecurity Specialist", "Ethical Hacking", "Penetration Testing"],
    },
    "U10": {
        "name": "Татьяна Козлова",
        "experience": 4,
        "position": "Бизнес-аналитик",
        "tags": ["Data & Analytics", "Data Analyst", "SQL", "Excel", "Business Intelligence"],
    }
}

# Подготовка данных
all_tags = set()
for course in courses:
    all_tags.update(course["tags"])
for user in users.values():
    all_tags.update(user["tags"])

tag_encoder = LabelEncoder()
tag_encoder.fit(list(all_tags))

course_tags = [tag_encoder.transform(course["tags"]) for course in courses]
user_interests = [tag_encoder.transform(user["tags"]) for user in users.values()]

max_len = max(
    max(len(tags) for tags in course_tags),
    max(len(interests) for interests in user_interests)
)

def pad_and_convert(data):
    padded = np.array([
        np.pad(arr, (0, max_len - len(arr)), mode='constant') 
        for arr in data
    ])
    return torch.tensor(padded, dtype=torch.long)

course_tensor = pad_and_convert(course_tags)
user_tensor = pad_and_convert(user_interests)

# Инициализация модели
num_tags = len(tag_encoder.classes_)
model = CourseRecommender(num_tags)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Обучение
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
        # Добавление шума
        score += np.random.uniform(-0.05, 0.05)
        row[user_id] = round(np.clip(score, 0, 1), 2)
    matrix.append(row)

# Преобразование матрицы в DataFrame
matrix_df = pd.DataFrame(matrix)

# Заполнение пропущенных значений нулями (если такие есть)
matrix_df = matrix_df.fillna(0)

# Преобразование DataFrame в numpy array, а затем в тензор
y_true = torch.FloatTensor(matrix_df.values)

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(course_tensor, user_tensor)
    loss = criterion(outputs, y_true)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Сохранение модели
save_model(model, "course_recommender.pth")

# 4. Получение рекомендаций
# --------------------------------------------------
model.eval()
with torch.no_grad():
    similarity_matrix = model(course_tensor, user_tensor)
    
    for user_idx, user in enumerate(users.values()):
        scores = similarity_matrix[:, user_idx]
        top_courses = torch.topk(scores, k=3).indices.tolist()
        
        print(f"\nРекомендации для {user['name']}:")
        for course_idx in top_courses:
            course = courses[course_idx]
            score = scores[course_idx].item()
            print(f"- {course['name']} (score: {score:.2f})")