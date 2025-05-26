import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from backend.database.User import User, Course, db
from backend.ml_model.scripts.question_to_tags import assign_user_tags

def export_users_to_dataframe():
    users = []
    for user in User.query.all():
        answers = user.preferences or {}
        tags_dict = assign_user_tags(answers)
        tags = tags_dict.get("user_id", {})
        all_tags = []
        for key, value in tags.items():
            if isinstance(value, list):
                all_tags.extend(value)
            elif value:
                all_tags.append(value)
        user_row = {
            'user_id': str(user.user_id),
            'preferred_tags': all_tags
        }
        users.append(user_row)
    df = pd.DataFrame(users)
    if 'preferred_tags' not in df.columns:
        df['preferred_tags'] = [[] for _ in range(len(df))]
    return df

def export_courses_to_dataframe():
    courses = []
    for course in Course.query.all():
        tags = course.tags if course.tags else []
        course_row = {
            'course_id': str(course.course_id),
            'tags': tags
        }
        courses.append(course_row)
    df = pd.DataFrame(courses)
    if 'tags' not in df.columns:
        df['tags'] = [[] for _ in range(len(df))]
    return df

def convert_tags_to_list(df, id_col):
    df_tags = df.copy()
    tag_cols = [col for col in df_tags.columns if col != id_col]
    df_tags['tags'] = df_tags[tag_cols].values.tolist()
    df_tags['tags'] = df_tags['tags'].apply(lambda tags: [str(tag) for tag in tags if tag and tag != ''])
    return df_tags[[id_col, 'tags']]

def convert_users_tags_to_list(df):
    df_tags = df.copy()
    tag_cols = [col for col in df_tags.columns if col != 'user_id']
    df_tags['preferred_tags'] = df_tags[tag_cols].values.tolist()
    df_tags['preferred_tags'] = df_tags['preferred_tags'].apply(lambda tags: [str(tag) for tag in tags if tag and tag != ''])
    return df_tags[['user_id', 'preferred_tags']]

def load_train_data():
    print("=" * 50)
    print("Загрузка тренировочных данных...")
    users_raw = pd.read_csv("backend/database/train_data/users.csv")
    courses_raw = pd.read_csv("backend/database/train_data/courses.csv")
    interactions = pd.read_csv("backend/database/train_data/interactions.csv")
    print(f"- Загружено пользователей: {len(users_raw)}")
    print(f"- Загружено курсов: {len(courses_raw)}")
    print(f"- Загружено взаимодействий: {len(interactions)}")
    print("=" * 50)

    users_tags = convert_users_tags_to_list(users_raw)
    courses_tags = convert_tags_to_list(courses_raw, "course_id")

    return users_raw, courses_raw, users_tags, courses_tags, interactions

def load_data(app):
    with app.app_context():
        print("=" * 50)
        print("Загрузка данных из БД...")
        users = []
        for user in User.query.all():
            answers = user.preferences or {}
            tags_dict = assign_user_tags(answers)
            tags = tags_dict.get("user_id", {})
            all_tags = []
            for key, value in tags.items():
                if isinstance(value, list):
                    all_tags.extend(value)
                elif value:
                    all_tags.append(value)
            user_row = {
                "user_id": str(user.user_id),
                "preferred_tags": all_tags
            }
            users.append(user_row)
        users_df = pd.DataFrame(users)
        if 'preferred_tags' not in users_df.columns:
            users_df['preferred_tags'] = [[] for _ in range(len(users_df))]

        courses = []
        for course in Course.query.all():
            tags = course.tags if course.tags else []
            course_row = {
                "course_id": str(course.course_id),
                "tags": tags
            }
            courses.append(course_row)
        courses_df = pd.DataFrame(courses)
        if 'tags' not in courses_df.columns:
            courses_df['tags'] = [[] for _ in range(len(courses_df))]
        interactions = []  # Если не нужны — оставь пустым
        print(users_df)
        print(courses_df)
        print(f"- Загружено пользователей: {len(users)}")
        print(f"- Загружено курсов: {len(courses)}")
        print(f"- Загружено взаимодействий: {len(interactions)}")
        print("=" * 50)
    return users_df, courses_df, interactions

def prepare_features(users, courses):
    mlb = MultiLabelBinarizer()
    all_tags = pd.Series(users['preferred_tags'].tolist() + courses['tags'].tolist())
    mlb.fit(all_tags)
    user_features = mlb.transform(users['preferred_tags'])
    course_features = mlb.transform(courses['tags'])
    print("Подготовка признаков...")
    print(f"- MultiLabelBinarizer применен к пользователям: размерность {user_features.shape[1]}")
    print(f"- MultiLabelBinarizer применен к курсам: размерность {course_features.shape[1]}")
    print("=" * 50)
    return user_features, course_features
