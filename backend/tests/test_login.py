import pytest
from backend.database.User import User, db

def test_login_success(client, test_user):
    # Тест успешного логина
    response = client.post('/login/', json={
        'email': 'test@example.com',
        'password': 'secret'
    })
    assert response.status_code == 200
    assert 'token' in response.json

def test_login_wrong_password(client, test_user):
    # Тест неверного пароля
    response = client.post('/login/', json={
        'email': 'test@example.com',
        'password': 'wrong'
    })
    assert response.status_code == 400
    assert response.json['message'] == 'Invalid password'

def test_login_user_not_found(client):
    # Тест несуществующего пользователя
    response = client.post('/login/', json={
        'email': 'nonexistent@example.com',
        'password': 'secret'
    })
    assert response.status_code == 400
    assert response.json['message'] == 'Invalid password'
