import pytest
from backend.app import create_app
from backend.database import db
import psycopg2
from sqlalchemy import create_engine

# Фикстура для создания и удаления тестовой базы данных
@pytest.fixture(scope='session', autouse=True)
def create_test_database():
    # Создаем тестовую базу данных
    conn = psycopg2.connect(
        dbname='postgres',  # Подключаемся к базе данных по умолчанию
        user='your_username',
        password='your_password',
        host='localhost',
        port='5432'
    )
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute('CREATE DATABASE test_db;')
    cursor.close()
    conn.close()

    # Настраиваем приложение для использования тестовой базы данных
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://your_username:your_password@localhost:5432/test_db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Создаем таблицы в тестовой базе данных
    with app.app_context():
        db.create_all()

    yield  # Тесты выполняются здесь

    # Удаляем тестовую базу данных после завершения тестов
    conn = psycopg2.connect(
        dbname='postgres',
        user='your_username',
        password='your_password',
        host='localhost',
        port='5432'
    )
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute('DROP DATABASE test_db;')
    cursor.close()
    conn.close()

# Фикстура для клиента тестов
@pytest.fixture(scope='module')
def client():
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://your_username:your_password@localhost:5432/test_db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    with app.test_client() as client:
        with app.app_context():
            yield client

# Фикстура для создания тестового пользователя
@pytest.fixture
def test_user(client):
    from backend.database.User import User, db

    user = User(
        first_name='John',
        last_name='Doe',
        email='test@example.com',
        date_of_birth='1990-01-01',
        job_position='Developer',
        work_experience=5,
        preferences={'theme': 'dark'}
    )
    user.set_password('secret')
    db.session.add(user)
    db.session.commit()
    return user
