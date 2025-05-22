# Создаем искусственную базу данных пользователей, чтобы обучать на них модель
import sqlite3


def populate_tags(cursor, tags):
    # Вставляем теги, игнорируя дубликаты
    cursor.executemany( 
        "INSERT OR IGNORE INTO tags (name) VALUES (?)",
        [(tag,) for tag in tags]
    )

def generate_combinations():
    from itertools import combinations

    # Создаем список всех вопросов (от 1 до 10)
    questions = list(range(1, 11))

    # Генерируем все возможные комбинации
    all_combinations = []
    for r in range(1, len(questions) + 1):
        all_combinations.extend(combinations(questions, r))

    return all_combinations

# Функция для получения уникальных тегов по выбранным пунктам
def get_unique_tags(tags_by_question, selected_questions):
    unique_tags = set()
    for question in selected_questions:
        unique_tags.update(tags_by_question[question])
    return sorted(unique_tags)  # Сортируем для удобства

def add_user(cursor, name, user_id, user_tags):
    cursor.execute("INSERT INTO users (name) VALUES (?)", (name, ))
    cursor.execute(
        "SELECT id FROM tags WHERE name IN ({})".format(
            ','.join(['?']*len(user_tags))
        ), 
        user_tags
    )
    tag_ids = [row[0] for row in cursor.fetchall()]

    cursor.executemany(
        "INSERT OR IGNORE INTO user_tags (user_id, tag_id) VALUES (?, ?)",
        [(user_id, tag_id) for tag_id in tag_ids]
    )

def main():
    # Подключаемся к БД
    conn = sqlite3.connect('backend/database/users.db')
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT
    )
    """)

    # Создаем таблицы
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_tags (
            user_id INTEGER,
            tag_id INTEGER,
            PRIMARY KEY (user_id, tag_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
        )
    ''')

    all_tags = ['Adobe XD', 'PHP', 'Python', 'Intermediate', 'Управление продуктом', 'Программирование на Python', 'Искусственный интеллект', 'CSS', 'Интернет-маркетолог', 'Graphic Designer', 'Java', 'C++', 'Финансовый анализ', 'Контекстная реклама', 'Game Developer', 'Docker', 'System Administrator', 'Unity', 'Figma', 'DevOps Engineer', 'Data Analyst', 'Интерьерный дизайн', 'Mobile Developer', 'IT Recruiter', 'Ручное тестирование', 'QA', 'Управление проектами', 'Data Scientist', 'SQL', 'Автоматизированное тестирование', 'Excel', 'Tableau', 'Adobe InDesign', 'Android', 'CI/CD', '.NET', 'Интенсив', 'Cinema 4D', 'Product Manager', 'Программирование на C#', '1С', 'Интерактивный дизайн', 'Project Manager', 'Penetration Testing', 'Marketing', 'Фотография', 'Backend Developer', 'TypeScript', 'Программирование', 'Машинное обучение', 'Machine Learning Engineer', 'Программирование на Java', 'Cybersecurity Specialist', 'Дизайн-системы', 'Нейронные сети', 'SEO', 'Linux', 'Видеомонтаж', 'Fullstack Developer', '3D-дизайн', 'Kotlin', 'Management', 'Веб-дизайн', 'Spring Framework', 'Django', 'Программирование на JavaScript', 'Beginner', 'Adobe Photoshop', 'HR', 'After Effects', 'Kubernetes', 'QA Engineer', 'Моушн-дизайн', 'Adobe Illustrator', 'Сети', 'Business Intelligence', 'HTML', 'Swift', 'React', 'C#', 'Design & UX', 'Frontend Developer', 'JavaScript', 'Blender', 'Cybersecurity', 'Рекрутинг', 'Development', 'Ethical Hacking', 'Advanced', 'UI/UX Designer', 'Data & Analytics']

    populate_tags(cursor, all_tags)

    tags_by_question = {
        1: ["Figma", "Adobe XD", "UI/UX Designer", "Веб-дизайн", "Интерактивный дизайн", "Adobe Photoshop", "Adobe Illustrator", "HTML", "CSS", "JavaScript", "React", "Design & UX"],
        2: ["Backend Developer", "Python", "Java", "C#", "PHP", "JavaScript", "Node.js", "Django", "Spring Framework", ".NET", "Программирование", "Программирование на Python", "Программирование на Java", "Программирование на C#", "Программирование на JavaScript"],
        3: ["SQL", "Data Analyst", "Data Scientist", "Business Intelligence", "Database Management", "Backend Developer"],
        4: ["Data Analyst", "Data Scientist", "Машинное обучение", "Искусственный интеллект", "Python", "Tableau", "Excel", "Business Intelligence", "Machine Learning Engineer", "Нейронные сети"],
        5: ["Fullstack Developer", "Frontend Developer", "Backend Developer", "JavaScript", "React", "Node.js", "REST API", "GraphQL", "Python", "Django", "Spring Framework"],
        6: ["Backend Developer", "Fullstack Developer", "Python", "JavaScript", "Node.js", "Django", "Spring Framework", "REST API", "GraphQL", "Программирование на Python", "Программирование на JavaScript"],
        7: ["Frontend Developer", "UI/UX Designer", "HTML", "CSS", "JavaScript", "React", "Веб-дизайн", "Интерактивный дизайн", "Figma", "Adobe XD"],
        8: ["Python", "JavaScript", "Automation", "DevOps Engineer", "CI/CD", "Docker", "Kubernetes", "Программирование на Python", "Программирование на JavaScript"],
        9: ["Data Scientist", "Machine Learning Engineer", "Машинное обучение", "Искусственный интеллект", "Python", "Нейронные сети", "Business Intelligence"],
        10: ["Frontend Developer", "Backend Developer", "Fullstack Developer", "JavaScript", "React", "Node.js", "Django", "Spring Framework", "CI/CD", "Docker", "Kubernetes", "DevOps Engineer", "Оптимизация производительности"]
    }

    all_combinations = generate_combinations()

    # Обрабатываем каждую комбинацию
    for i, combination in enumerate(all_combinations, 1):
        unique_tags = get_unique_tags(tags_by_question, combination)
        add_user(cursor, 'U' + f'{i}', i, unique_tags)
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()