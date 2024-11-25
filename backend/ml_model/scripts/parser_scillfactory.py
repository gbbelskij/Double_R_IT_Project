import requests
from bs4 import BeautifulSoup
import sqlite3

url = "https://skillfactory.ru/courses"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


conn = sqlite3.connect("../database/courses.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    link TEXT,
    duration TEXT,
    description TEXT,
    price TEXT,
    type TEXT,
    direction TEXT
)
""")
conn.commit()

response = requests.get(url, headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    courses = soup.find_all('li', class_='card')

    for course in courses:
        # Название курса
        title_tag = course.find('h3', class_='card__title')
        title = title_tag.text.strip() if title_tag else "Название отсутствует"

        # Ссылка на курс
        link_tag = title_tag.find(
            'a', class_='title__link') if title_tag else None
        link = link_tag['href'] if link_tag else "Ссылка отсутствует"

        # Длительность курса
        duration_tag = course.find('p', class_='card__duration')
        duration = duration_tag.text.strip() if duration_tag else "Длительность отсутствует"

        # Описание курса
        description_tag = course.find('p', class_='card__description')
        description = description_tag.text.strip(
        ) if description_tag else "Описание отсутствует"

        # Цена курса
        price_tag = course.find('p', class_='card__current-price')
        price = price_tag.text.strip() if price_tag else "Цена отсутствует"

        # Тип курса
        type_tag = course.find_all('p', class_='card__type')
        type = type_tag[0].text

        # Направление
        direction_tag = course.find_all('p', class_='card__type')
        direction = direction_tag[1].text

        if (type in ["Профессия", "Интенсив", "Курс", "Специализация"]):
            print(f"Название: {title}")
            print(f"Ссылка: {link}")
            print(f"Длительность: {duration}")
            print(f"Описание: {description}")
            print(f"Цена: {price}")
            print(f"Тип: {type}")
            print(f"Направление: {direction}")
            print("-" * 40)

            cursor.execute("""
            INSERT INTO courses (title, link, duration, description, price, type, direction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (title, link, duration, description, price, type, direction))

    conn.commit()
else:
    print(f"Ошибка при запросе страницы: {response.status_code}")

conn.close()
