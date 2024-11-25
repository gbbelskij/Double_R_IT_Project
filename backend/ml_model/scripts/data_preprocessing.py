import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def main():
    input_file = 'ml_model/data/raw/raw_test_data.csv'
    columns_names_file = 'ml_model/data/preprocessed/columns_names.csv'
    output_file = 'ml_model/data/preprocessed/normalized_test_data.csv'

    data = pd.read_csv(input_file)
    output_columns = pd.read_csv(columns_names_file).columns

    scaler = MinMaxScaler()
    numeric_columns = data[['Age', 'Experience']]
    normalized_numeric = scaler.fit_transform(numeric_columns)

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


if __name__ == "__main__":
    main()
