import sqlite3


# Теги курсов, разбитые на них  самих (всего курсов 79 штук пока)
course_tags = [
    # ID 1
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Adobe Photoshop', 'Adobe Illustrator'],
    # ID 2
    ['Intermediate', 'QA', 'QA Engineer', 'Python', 'Автоматизированное тестирование'],
    # ID 3
    ['Beginner', 'Design & UX', 'UI/UX Designer', 'Figma', 'Adobe XD'],
    # ID 4
    ['Intermediate', 'Cybersecurity', 'Cybersecurity Specialist', 'Ethical Hacking', 'Penetration Testing'],
    # ID 5
    ['Intermediate', 'Design & UX', 'Graphic Designer', '3D-дизайн', 'Blender', 'Cinema 4D'],
    # ID 6
    ['Beginner', 'Development', 'Backend Developer', '1С', 'Программирование на C#'],
    # ID 7
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Интерьерный дизайн'],
    # ID 8
    ['Intermediate', 'Development', 'Backend Developer', 'Java', 'Spring Framework'],
    # ID 9
    ['Beginner', 'Design & UX', 'Graphic Designer', 'After Effects', 'Моушн-дизайн'],
    # ID 10
    ['Intermediate', 'Development', 'Backend Developer', 'C++', 'Программирование'],
    # ID 11
    ['Beginner', 'Design & UX', 'Game Developer', 'Unity', '3D-дизайн'],
    # ID 12
    ['Beginner', 'Development', 'Backend Developer', 'Python'],
    # ID 13
    ['Advanced', 'Development', 'Backend Developer', 'Python'],
    # ID 14
    ['Beginner', 'Design & UX', 'UI/UX Designer', 'Figma', 'Веб-дизайн'],
    # ID 15
    ['Beginner', 'Data & Analytics', 'Data Analyst', 'SQL', 'Excel', 'Tableau'],
    # ID 16
    ['Intermediate', 'Data & Analytics', 'Data Scientist', 'Python', 'Машинное обучение'],
    # ID 17
    ['Advanced', 'Data & Analytics', 'Data Scientist', 'Python', 'Машинное обучение', 'Нейронные сети'],
    # ID 18
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Adobe Illustrator'],
    # ID 19
    ['Beginner', 'Development', 'System Administrator', 'Linux', 'Сети'],
    # ID 20
    ['Intermediate', 'Development', 'Fullstack Developer', 'Python', 'JavaScript', 'Django'],
    # ID 21
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Figma', 'Adobe Photoshop'],
    # ID 22
    ['Beginner', 'Development', 'Game Developer', 'Unity', 'C#'],
    # ID 23
    ['Advanced', 'Development', 'Game Developer', 'Unity', 'C#'],
    # ID 24
    ['Intermediate', 'Development', 'Mobile Developer', 'Java', 'Kotlin'],
    # ID 25
    ['Beginner', 'Development', 'Frontend Developer', 'JavaScript', 'React'],
    # ID 26
    ['Advanced', 'Development', 'Frontend Developer', 'JavaScript', 'React', 'TypeScript'],
    # ID 27
    ['Advanced', 'Data & Analytics', 'Machine Learning Engineer', 'Нейронные сети', 'Python'],
    # ID 28
    ['Intermediate', 'Design & UX', 'Graphic Designer', 'Нейронные сети', 'Искусственный интеллект'],
    # ID 29
    ['Beginner', 'QA', 'QA Engineer', 'Ручное тестирование'],
    # ID 30
    ['Intermediate', 'Development', 'Backend Developer', 'C#', '.NET'],
    # ID 31
    ['Beginner', 'Design & UX', 'Graphic Designer', 'After Effects', 'Видеомонтаж'],
    # ID 32
    ['Beginner', 'Data & Analytics', 'Data Analyst', 'SQL'],
    # ID 33
    ['Intermediate', 'Data & Analytics', 'Data Analyst', 'Excel', 'Финансовый анализ'],
    # ID 34
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Adobe Photoshop', 'Adobe Illustrator'],
    # ID 35
    ['Intermediate', 'Management', 'Project Manager', 'Управление проектами'],
    # ID 36
    ['Beginner', 'Marketing', 'Интернет-маркетолог', 'SEO', 'Контекстная реклама'],
    # ID 37
    ['Intermediate', 'Management', 'Product Manager', 'Управление продуктом'],
    # ID 38
    ['Beginner', 'Design & UX', 'UI/UX Designer', 'Figma', 'Adobe XD'],
    # ID 39
    ['Beginner', 'Development', 'Backend Developer', 'Python', 'Интенсив'],
    # ID 40
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Adobe Photoshop', 'Интенсив'],
    # ID 41
    ['Intermediate', 'Design & UX', 'Graphic Designer', 'After Effects', '3D-дизайн'],
    # ID 42
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Интерьерный дизайн'],
    # ID 43
    ['Beginner', 'Development', 'Fullstack Developer', 'HTML', 'CSS', 'JavaScript'],
    # ID 44
    ['Intermediate', 'Development', 'Fullstack Developer', 'PHP', 'JavaScript'],
    # ID 45
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Фотография'],
    # ID 46
    ['Beginner', 'HR', 'IT Recruiter', 'Рекрутинг'],
    # ID 47
    ['Intermediate', 'Development', 'Backend Developer', 'Java', 'Spring Framework'],
    # ID 48
    ['Advanced', 'Data & Analytics', 'Machine Learning Engineer', 'Нейронные сети', 'Python'],
    # ID 49
    ['Beginner', 'Development', 'Mobile Developer', 'Swift'],
    # ID 50
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Adobe Photoshop', 'Adobe Illustrator'],
    # ID 51
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Интерьерный дизайн'],
    # ID 52
    ['Intermediate', 'Design & UX', 'Graphic Designer', '3D-дизайн', 'Blender', 'Cinema 4D'],
    # ID 53
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Figma', 'Adobe Photoshop'],
    # ID 54
    ['Beginner', 'Design & UX', 'UI/UX Designer', 'Figma', 'Adobe XD'],
    # ID 55
    ['Beginner', 'Design & UX', 'Game Developer', 'Unity', '3D-дизайн'],
    # ID 56
    ['Beginner', 'Design & UX', 'Graphic Designer', 'After Effects', 'Моушн-дизайн'],
    # ID 57
    ['Beginner', 'Design & UX', 'UI/UX Designer', 'Figma', 'Веб-дизайн'],
    # ID 58
    ['Intermediate', 'Design & UX', 'Graphic Designer', 'Нейронные сети', 'Искусственный интеллект'],
    # ID 59
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Adobe Illustrator'],
    # ID 60
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Интерьерный дизайн'],
    # ID 61
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Adobe Photoshop', 'Adobe Illustrator'],
    # ID 62
    ['Beginner', 'Design & UX', 'UI/UX Designer', 'Figma', 'Adobe XD'],
    # ID 63
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Adobe Photoshop', 'Интенсив'],
    # ID 64
    ['Intermediate', 'Design & UX', 'Graphic Designer', 'After Effects', '3D-дизайн'],
    # ID 65
    ['Beginner', 'Design & UX', 'Graphic Designer', 'After Effects', 'Видеомонтаж'],
    # ID 66
    ['Intermediate', 'Design & UX', 'Graphic Designer', 'Интерактивный дизайн'],
    # ID 67
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Фотография'],
    # ID 68
    ['Intermediate', 'Design & UX', 'UI/UX Designer', 'Figma', 'Дизайн-системы'],
    # ID 69
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Adobe Photoshop'],
    # ID 70
    ['Beginner', 'Design & UX', 'UI/UX Designer', 'Figma'],
    # ID 71
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Adobe Illustrator'],
    # ID 72
    ['Intermediate', 'Design & UX', 'Graphic Designer', 'After Effects'],
    # ID 73
    ['Intermediate', 'Design & UX', 'Graphic Designer', 'Cinema 4D'],
    # ID 74
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Blender'],
    # ID 75
    ['Beginner', 'Design & UX', 'Graphic Designer', 'Adobe InDesign'],
    # ID 76
    ['Beginner', 'Development', 'Backend Developer', 'Python', 'Интенсив'],
    # ID 77
    ['Beginner', 'QA', 'QA Engineer', 'Ручное тестирование', 'Интенсив'],
    # ID 78
    ['Beginner', 'Development', 'Frontend Developer', 'JavaScript', 'Интенсив'],
    # ID 79
    ['Beginner', 'Data & Analytics', 'Data Scientist', 'Python', 'Машинное обучение'],
]

def populate_tags(cursor, tags):
    # Вставляем теги, игнорируя дубликаты
    cursor.executemany( 
        "INSERT OR IGNORE INTO tags (name) VALUES (?)",
        [(tag,) for tag in tags]
    )

def populate_course_tags(cursor, course_tags):
    # Для каждого курса и его тегов
    for course_id, tags in enumerate(course_tags, start=1):
        # Получаем ID всех тегов курса
        cursor.execute(
            "SELECT id FROM tags WHERE name IN ({})".format(
                ','.join(['?']*len(tags))
            ), 
            tags
        )
        tag_ids = [row[0] for row in cursor.fetchall()]
        
        # Связываем курс с тегами
        cursor.executemany(
            "INSERT OR IGNORE INTO course_tags (course_id, tag_id) VALUES (?, ?)",
            [(course_id, tag_id) for tag_id in tag_ids]
        )

def main():
    # Подключаемся к БД
    conn = sqlite3.connect('backend/database/courses.db')
    cursor = conn.cursor()
    
    # Создаем таблицы
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS course_tags (
            course_id INTEGER,
            tag_id INTEGER,
            PRIMARY KEY (course_id, tag_id),
            FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
        )
    ''')
    
    # Собираем все уникальные теги
    all_tags = set()
    for tags in course_tags:
        all_tags.update(tags)
    
    # Заполняем таблицы
    populate_tags(cursor, all_tags)
    populate_course_tags(cursor, course_tags)
    
    # Сохраняем изменения и закрываем соединение
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()