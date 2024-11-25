import requests
from bs4 import BeautifulSoup
import sqlite3

URL = "https://skillfactory.ru/courses"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def create_database():
    connection = sqlite3.connect("../database/courses.db")
    cursor = connection.cursor()
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
    connection.commit()
    return connection, cursor

def fetch_courses():
    response = requests.get(URL, headers=HEADERS)
    if response.status_code != 200:
        print(f"Error fetching the page: {response.status_code}")
        return None
    return BeautifulSoup(response.text, 'html.parser')

def extract_course_data(course):
    title_tag = course.find('h3', class_='card__title')
    title = title_tag.text.strip() if title_tag else "Title not available"
    
    link_tag = title_tag.find('a', class_='title__link') if title_tag else None
    link = link_tag['href'] if link_tag else "Link not available"

    duration_tag = course.find('p', class_='card__duration')
    duration = duration_tag.text.strip() if duration_tag else "Duration not available"

    description_tag = course.find('p', class_='card__description')
    description = description_tag.text.strip() if description_tag else "Description not available"

    price_tag = course.find('p', class_='card__current-price')
    price = price_tag.text.strip() if price_tag else "Price not available"

    type_tag = course.find_all('p', class_='card__type')
    course_type = type_tag[0].text if type_tag else "Type not available"

    direction_tag = course.find_all('p', class_='card__type')
    direction = direction_tag[1].text if len(direction_tag) > 1 else "Direction not available"

    return title, link, duration, description, price, course_type, direction

def main():
    connection, cursor = create_database()
    soup = fetch_courses()
    if soup is None:
        return

    courses = soup.find_all('li', class_='card')
    for course in courses:
        title, link, duration, description, price, course_type, direction = extract_course_data(course)
        
        if course_type in ["Профессия", "Интенсив", "Курс", "Специализация"]:
            cursor.execute("""
            INSERT INTO courses (title, link, duration, description, price, type, direction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (title, link, duration, description, price, course_type, direction))

    connection.commit()
    connection.close()

if __name__ == "__main__":
    main()

