import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_data():
    users = pd.read_csv("backend/database/train_data/users.csv")
    courses = pd.read_csv("backend/database/train_data/courses.csv")
    interactions = pd.read_csv("backend/database/train_data/interactions.csv")
    return users, courses, interactions

def prepare_features(users, courses):
    encoder = OneHotEncoder()
    user_features = encoder.fit_transform(users.iloc[:, 1:]).toarray()
    course_features = encoder.fit_transform(courses.iloc[:, 1:]).toarray()
    return user_features, course_features
