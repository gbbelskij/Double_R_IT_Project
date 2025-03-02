import sqlite3

course_tags = [
    ['beginner', 'graphic-design', 'adobe'],
    ['beginner', 'python', 'software-testing', 'automation'],
    ['beginner', 'ui-ux', 'web-design', 'prototyping'],
    ['intermediate', 'cybersecurity', 'ethical-hacking'],
    ['beginner', '3d-modeling', 'animation', 'blender'],
    ['beginner', '1c', 'business-automation'],
    ['beginner', 'interior-design', 'space-planning'],
    ['beginner', 'java', 'spring', 'web-development', 'devops'],
    ['beginner', 'motion-design', '2d-animation', '3d-animation'],
    ['beginner', 'cpp', 'software-development'],
    ['beginner', 'game-design', 'game-development'],
    ['beginner', 'python', 'web-development', 'backend'],
    ['intermediate', 'python', 'software-development'],
    ['beginner', 'web-design', 'ui-ux', 'graphic-design'],
    ['beginner', 'data-analysis', 'business-intelligence'],
    ['intermediate', 'data-science', 'machine-learning'],
    ['advanced', 'data-science', 'machine-learning', 'mathematics'],
    ['beginner', 'illustration', 'branding', 'fashion-design'],
    ['beginner', 'linux', 'system-administration'],
    ['intermediate', 'python', 'fullstack', 'web-development'],
    ['beginner', 'graphic-design', 'career-development'],
    ['beginner', 'csharp', 'game-development', 'unity'],
    ['intermediate', 'csharp', 'game-development', 'unity', '3d-modeling'],
    ['beginner', 'android', 'mobile-development'],
    ['beginner', 'javascript', 'react', 'web-development', 'frontend'],
    ['advanced', 'javascript', 'typescript', 'react', 'frontend'],
    ['advanced', 'neural-networks', 'deep-learning', 'machine-learning'],
    ['beginner', 'neural-networks', 'creative-tools'],
    ['beginner', 'software-testing', 'qa'],
    ['intermediate', 'csharp', 'dotnet', 'web-development'],
    ['beginner', 'video-editing', 'motion-design'],
    ['beginner', 'sql', 'data-analysis'],
    ['intermediate', 'finance', 'risk-management'],
    ['beginner', 'graphic-design', 'adobe'],
    ['intermediate', 'project-management', 'team-leadership'],
    ['beginner', 'digital-marketing', 'seo', 'social-media'],
    ['intermediate', 'product-management', 'agile'],
    ['beginner', 'ui-ux', 'web-design'],
    ['beginner', 'python', 'web-development'],
    ['beginner', 'graphic-design', 'adobe'],
    ['beginner', 'motion-design', '3d-animation'],
    ['beginner', 'interior-design', 'space-planning'],
    ['beginner', 'web-development', 'frontend', 'backend'],
    ['intermediate', 'php', 'javascript', 'fullstack'],
    ['beginner', 'photography', 'creative-arts'],
    ['beginner', 'recruitment', 'hr'],
    ['intermediate', 'java', 'spring', 'web-development'],
    ['advanced', 'machine-learning', 'neural-networks'],
    ['beginner', 'ios', 'swift', 'mobile-development'],
    ['beginner', 'graphic-design', 'adobe'],
    ['beginner', 'interior-design', 'space-planning'],
    ['beginner', '3d-modeling', 'animation', 'blender'],
    ['beginner', 'graphic-design', 'career-development'],
    ['advanced', 'ui-ux', 'web-design'],
    ['beginner', 'game-design', 'game-development'],
    ['advanced', 'motion-design', '3d-animation'],
    ['beginner', 'web-design', 'ui-ux'],
    ['beginner', 'neural-networks', 'creative-tools'],
    ['beginner', 'illustration', 'branding'],
    ['beginner', 'interior-design', 'space-planning'],
    ['beginner', 'graphic-design', 'adobe'],
    ['beginner', 'ui-ux', 'web-design'],
    ['beginner', 'graphic-design', 'adobe'],
    ['beginner', 'motion-design', '3d-animation'],
    ['beginner', 'video-editing', 'motion-design'],
    ['intermediate', 'interactive-media', 'multimedia'],
    ['beginner', 'photography', 'creative-arts'],
    ['intermediate', 'design-systems', 'ux'],
    ['beginner', 'adobe-photoshop', 'graphic-design'],
    ['beginner', 'figma', 'ui-ux'],
    ['beginner', 'adobe-illustrator', 'vector-graphics'],
    ['beginner', 'after-effects', 'motion-design'],
    ['beginner', 'cinema-4d', '3d-modeling'],
    ['beginner', 'blender', '3d-modeling'],
    ['beginner', 'indesign', 'layout-design'],
    ['beginner', 'python', 'web-development'],
    ['beginner', 'software-testing', 'qa'],
    ['beginner', 'javascript', 'frontend', 'web-development'],
    ['beginner', 'data-science', 'machine-learning']
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