import pytest
from backend.database.User import User, db

def test_register_success(client):
    # Тест успешной регистрации
    response = client.post('/register/', json={
        'first_name': 'Jane',
        'last_name': 'Doe',
        'email': 'jane@example.com',
        'password': 'secret',
        'date_of_birth': '1995-01-01',
        'job_position': 'Designer',
        'work_experience': 3,
        'preferences': {'theme': 'light'}
    })
    assert response.status_code == 201
    assert response.json['message'] == 'User registered successfully'

def test_register_duplicate_email(client, test_user):
    # Тест регистрации с уже занятым email
    response = client.post('/register/', json={
        'first_name': 'John',
        'last_name': 'Doe',
        'email': 'test@example.com',
        'password': 'secret'
    })
    assert response.status_code == 400
    assert response.json['message'] == 'Email already taken'

def test_register_missing_fields(client):
    # Тест регистрации с отсутствующими обязательными полями
    response = client.post('/register/', json={
        'first_name': 'John',
        'email': 'test@example.com'
    })
    assert response.status_code == 400
    assert response.json['message'] == 'Missing required fields'
