import pandas as pd
import numpy as np
import json
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler

INPUT_FILE = 'backend/ml_model/data/raw/raw_test_data.csv'
OUTPUT_FILE = 'backend/ml_model/data/preprocessed/normalized_test_data.csv'
SCALER_SAVE_PATH = 'backend/ml_model/data/scaler/scaler.pkl'

def dp_from_csv(input_file, output_file):
    columns_names_file = 'backend/ml_model/data/preprocessed/columns_names.csv'
    
    data = pd.read_csv(input_file)
    output_columns = pd.read_csv(columns_names_file).columns

    scaler = MinMaxScaler()
    numeric_columns = data[['Age', 'Experience']]
    normalized_numeric = scaler.fit_transform(numeric_columns)
    dump(scaler, SCALER_SAVE_PATH)

    selected_role = data['Role']
    selected_department = data['Department']
    normilized_role_dep = np.array(
        [[0 for i in range(len(output_columns) - 12)] for j in range(len(data))])

    for i in range(len(data)):
        for j, column_name in enumerate(output_columns[2:28]):
            if selected_role[i] == column_name or selected_department[i] == column_name:
                normilized_role_dep[i][j] = 1

    answers = data['Answers'].apply(lambda x: np.array(eval(x)))
    answers_array = np.vstack(answers.values)

    expected_courses = np.array([[int(course_num)]
                                for course_num in data['Course']])

    processed_data = np.hstack((
        normalized_numeric,
        normilized_role_dep,
        answers_array,
        expected_courses
    ))

    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(output_file, index=False)

    print(f"Данные успешно нормализованы и сохранены в файл: {output_file}")

def dp_from_json(json_data: str) -> str:
    columns_names_file = 'backend/ml_model/data/preprocessed/columns_names.csv'

    data = json.loads(json_data)
    data = pd.DataFrame(data)
    output_columns = pd.read_csv(columns_names_file).columns

    scaler = load(SCALER_SAVE_PATH)
    numeric_columns = data[['Age', 'Experience']]
    normalized_numeric = scaler.transform(numeric_columns)

    selected_role = data['Role']
    selected_department = data['Department']
    normilized_role_dep = np.array(
        [[0 for i in range(len(output_columns) - 12)]])

    for j, column_name in enumerate(output_columns[2:28]):
        if selected_role[0] == column_name or selected_department[0] == column_name:
            normilized_role_dep[0][j] = 1

    answers = data['Answers'].apply(lambda x: np.array(eval(x)))
    answers_array = np.vstack(answers.values)

    expected_courses = np.array([[int(course_num)]
                                for course_num in data['Course']])

    processed_data = np.hstack((
        normalized_numeric,
        normilized_role_dep,
        answers_array,
        expected_courses
    ))

    processed_df = pd.DataFrame(processed_data)
    json_data = processed_df.to_json(index=False)
    
    return json_data
