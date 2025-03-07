# train.py
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import numpy as np
from model import CourseRecommender
from save_load import save_model
from matrix_create import matrix_create, get_users
import sqlite3

def get_courses():
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

    return courses

courses = get_courses()

# Пользователи и их интересы
users = get_users()

# Подготовка данных
def data_preproc():
    all_tags = set()
    for course in courses:
        all_tags.update(course["tags"])
    for user in users.values():
        all_tags.update(user)

    tag_encoder = LabelEncoder()
    tag_encoder.fit(list(all_tags))

    course_tags = [tag_encoder.transform(course["tags"]) for course in courses]
    user_interests = [tag_encoder.transform(list(user)) for user in users.values()]

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

    return (course_tensor, user_tensor, tag_encoder)

course_tensor, user_tensor, tag_encoder = data_preproc()

# Инициализация модели
num_tags = len(tag_encoder.classes_)
model = CourseRecommender(num_tags)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Обучение
# Генерация матрицы
def matrix_gen():
    matrix = matrix_create()

    # Преобразование матрицы в DataFrame
    matrix_df = pd.DataFrame(matrix)

    # Заполнение пропущенных значений нулями (если такие есть)
    matrix_df = matrix_df.fillna(0)

    # Преобразование DataFrame в numpy array, а затем в тензор
    y_true = torch.FloatTensor(matrix_df.values)

    return y_true

y_true = matrix_gen()

def train_model():
    for epoch in range(200):
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

train_model()

def test_recomendations():
    model.eval()
    with torch.no_grad():
        similarity_matrix = model(course_tensor, user_tensor)
        
        for user_idx, user in enumerate(users.values()):
            if user_idx == 5:
                break
            scores = similarity_matrix[:, user_idx]
            top_courses = torch.topk(scores, k=3).indices.tolist()
            
            print(f"\nРекомендации для {user_idx}:")
            print(f'USER TAGS: {user}')
            for course_idx in top_courses:
                course = courses[course_idx]
                score = scores[course_idx].item()
                print(f"- {course['name']} (score: {score:.2f})")

test_recomendations()