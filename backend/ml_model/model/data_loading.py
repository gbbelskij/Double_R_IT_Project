import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os
import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../database')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

import pandas as pd
import json
from backend.database.User import db, User, Course

from backend.ml_model.scripts.question_to_tags import assign_user_tags

def load_data(app):
    with app.app_context():
        # --- Загрузка пользователей ---
        users = []
        for user in User.query.all():
            # preferences — это строка или dict с ответами на вопросы
            answers = user.preferences
            if answers is None:
                answers = {}
            tags_dict = assign_user_tags(answers)
            tags = tags_dict.get("user_id", {})
            # Преобразуем списки тегов в строки для OneHotEncoder
            user_row = {"user_id": str(user.user_id)}
            for key, value in tags.items():
                if isinstance(value, list):
                    user_row[key] = ",".join(value)
                else:
                    user_row[key] = value
            users.append(user_row)
        # print(users)
        users_df = pd.DataFrame(users)

        # --- Загрузка курсов ---
        courses = []
        for course in Course.query.all():
            # Если есть course.preferences — аналогично, иначе просто course_id
            course_row = {"course_id": str(course.course_id)}
            # Можно добавить дополнительные признаки, если нужны
            courses.append(course_row)
        courses_df = pd.DataFrame(courses)

        # --- Загрузка взаимодействий ---
        interactions = []
        # for interaction in Interaction.query.all():
        #     interactions.append({
        #         "user_id": str(interaction.user_id),
        #         "course_id": str(interaction.course_id),
        #         "liked": int(getattr(interaction, "liked", 1))  # если liked нет, считаем 1
        #     })
        interactions_df = pd.DataFrame(interactions)

        return users_df, courses_df, interactions_df

def prepare_features(users, courses):
    # Кодируем категориальные признаки (теги) в one-hot
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder()
    user_features = encoder.fit_transform(users.drop("user_id", axis=1).fillna(""))
    course_features = encoder.fit_transform(courses.drop("course_id", axis=1).fillna(""))
    return user_features, course_features
