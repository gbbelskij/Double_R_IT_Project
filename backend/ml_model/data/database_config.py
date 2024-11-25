import pandas as pd
import sqlite3

RAW_PATH = 'ml_model/data/raw/raw_test_data.csv'
PREPROCESSED_PATH = 'ml_model/data/preprocessed/normalized_test_data.csv'

# TODO: реализовать взаимодействие с файлами, получение различных выборок и тд

def get_test_data():
    data = pd.read_csv(PREPROCESSED_PATH, dtype=float)
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    return features, labels
    
def get_course(id):
    conn = sqlite3.connect("../database/courses.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM courses WHERE id = ?", (id))
    course = cursor.fetchone()
    conn.close()
    return course
