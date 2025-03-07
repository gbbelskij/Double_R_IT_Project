import sqlite3


def get_users():
    conn = sqlite3.connect('backend/database/users.db')
    cursor = conn.cursor()

    # Запрос для получения user_id и связанных с ним тегов
    cursor.execute('''
        SELECT users.id, tags.name
        FROM users
        JOIN user_tags ON users.id = user_tags.user_id
        JOIN tags ON user_tags.tag_id = tags.id
    ''')

    # Обработка результатов для пользователей
    user_tags = {}
    for user_id, tag_name in cursor.fetchall():
        if user_id not in user_tags:
            user_tags[user_id] = set()
        user_tags[user_id].add(tag_name)

    # Закрытие соединения с базой данных
    conn.close()

    return user_tags

def get_courses():
    # Подключаемся к БД
    conn = sqlite3.connect('backend/database/courses.db')
    cursor = conn.cursor()

    # Запрос для получения course_id и связанных с ним тегов
    cursor.execute('''
        SELECT courses.id, tags.name
        FROM courses
        JOIN course_tags ON courses.id = course_tags.course_id
        JOIN tags ON course_tags.tag_id = tags.id
    ''')

    # Обработка результатов для курсов
    course_tags = {}
    for course_id, tag_name in cursor.fetchall():
        if course_id not in course_tags:
            course_tags[course_id] = set()
        course_tags[course_id].add(tag_name)

    conn.close()
    
    return course_tags

def matrix_create():
    users = get_users()
    courses = get_courses()

    matrix = []

    for course_tags in courses.values():
        course = []
        for user_tags in users.values():
            course_rate = 0.0
            intersection = course_tags & user_tags
            course_rate += len(intersection) * 0.1
            course.append(course_rate)
        matrix.append(course)

    return matrix

def main():
    print(matrix_create())

if __name__ == "__main__":
    main()