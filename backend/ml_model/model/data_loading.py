import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os
import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../database')))
# from User import db, User, Course, Interaction

def export_users_to_dataframe():
    users = User.query.all()
    
    data = []
    for user in users:
        preferences = user.preferences or {}
        data.append({
            'user_id': user.user_id,
            'Lang': preferences.get('Lang', ''),
            'Department': preferences.get('Department', ''),
            'Add tags': preferences.get('Add tags', ''),
            'Lvl': preferences.get('Lv1', '')
        })
    
    df = pd.DataFrame(data)
    
    return df

def export_courses_to_dataframe():
    courses = Course.query.all()
    
    data = []
    for course in courses:
        preferences = course.preferences or {}
        data.append({
            'course_id': course.course_id,
            'Lang': preferences.get('Lang', ''),
            'Department': preferences.get('Department', ''),
            'Add tags': preferences.get('Add tags', ''),
            'Lvl': preferences.get('Lv1', '')
        })
    
    df = pd.DataFrame(data)
    
    return df

def export_courses_to_dataframe():
    courses = Course.query.all()
    
    data = []
    for course in courses:
        preferences = course.preferences or {}
        data.append({
            'course_id': course.course_id,
            'Lang': preferences.get('Lang', ''),
            'Department': preferences.get('Department', ''),
            'Add tags': preferences.get('Add tags', ''),
            'Lvl': preferences.get('Lv1', '')
        })
    
    df = pd.DataFrame(data)
    
    return df

def export_interactions_to_dataframe():
    interactions = Interaction.query.all()
    
    data = []
    for interaction in interactions:
        data.append({
            'user_id': interaction.user_id,
            'course_id': interaction.course_id,
            'liked': 1 if interaction.liked else 0
        })
    
    df = pd.DataFrame(data)
    
    return df

def load_train_data():
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

def load_data():
    print("=" * 50)
    print("Загрузка данных...")
    users = pd.read_csv("backend/database/actual_data/users.csv")
    courses = pd.read_csv("backend/database/actual_data/courses.csv")
    interactions = pd.read_csv("backend/database/actual_data/interactions.csv")
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
