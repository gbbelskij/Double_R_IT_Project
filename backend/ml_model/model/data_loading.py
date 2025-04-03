import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_data():
    users = pd.read_csv("backend/database/train_data/users.csv")
    courses = pd.read_csv("backend/database/train_data/courses.csv")
    interactions = pd.read_csv("backend/database/train_data/interactions.csv")
    print("=" * 50)
    print("Загрузка данных...")
    users = pd.read_csv("backend/database/train_data/users.csv")
    courses = pd.read_csv("backend/database/train_data/courses.csv")
    interactions = pd.read_csv("backend/database/train_data/interactions.csv")
    print(f"- Загружено пользователей: {len(users)}")
    print(f"- Загружено курсов: {len(courses)}")
    print(f"- Загружено взаимодействий: {len(interactions)}")
    print("=" * 50)
    return users, courses, interactions

def prepare_features(users, courses):
    encoder = OneHotEncoder()
    user_features = encoder.fit_transform(users.iloc[:, 1:]).toarray()
    course_features = encoder.fit_transform(courses.iloc[:, 1:]).toarray()
    print("Подготовка признаков...")
    encoder = OneHotEncoder()
    user_features = encoder.fit_transform(users.iloc[:, 1:]).toarray()
    print(f"- OneHotEncoder применен к пользователям: размерность {user_features.shape[1]}")
    course_features = encoder.fit_transform(courses.iloc[:, 1:]).toarray()
    print(f"- OneHotEncoder применен к курсам: размерность {course_features.shape[1]}")
    print("=" * 50)
    return user_features, course_features
