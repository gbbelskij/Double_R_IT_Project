import pandas as pd
import numpy as np
import sqlite3

raw_path = ''
preprocessed_path = ''

# TODO: реализовать взаимодействие с файлами, получение различных выборок и тд

def get_course(id):
    conn = sqlite3.connect("../database/courses.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM courses WHERE id = ?", (id))
    course = cursor.fetchone()
    conn.close()
    return course

