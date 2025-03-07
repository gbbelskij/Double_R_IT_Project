# Проверка тегов по базе данных (что все теги привязались корректно)
import sqlite3

def get_all_courses_with_tags():
    # Подключаемся к БД
    conn = sqlite3.connect('backend/database/users.db')
    cursor = conn.cursor()

    # SQL-запрос для получения всех курсов и их тегов
    query = '''
        SELECT u.name, GROUP_CONCAT(t.name, ', ') AS tags
        FROM users u
        JOIN user_tags ut ON u.id = ut.user_id
        JOIN tags t ON ut.tag_id = t.id
        GROUP BY u.id
    '''

    # Выполняем запрос
    cursor.execute(query)
    results = cursor.fetchall()
    
    # Закрываем соединение
    conn.close()
    
    # Выводим результаты
    for user_name, tags in results[:1]:
        print(f"User: {user_name}")
        print(f"Tags: {tags}")
        print("-" * 40)

# Вызов функции
get_all_courses_with_tags()