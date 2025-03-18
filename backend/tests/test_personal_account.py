import pytest
from backend.database.User import User, db, TokenBlockList

def test_get_personal_account_valid_token(client, test_user):
    # Логин для получения токена
    login_response = client.post('/login/', json={
        'email': 'test@example.com',
        'password': 'secret'
    })
    token = login_response.json['token']

    # Тест получения данных пользователя
    response = client.get('/personal_account/', headers={'Authorization': f'Bearer {token}'})
    assert response.status_code == 200
    assert response.json['user_data']['email'] == 'test@example.com'

def test_update_personal_account(client, test_user):
    # Логин для получения токена
    login_response = client.post('/login/', json={
        'email': 'test@example.com',
        'password': 'secret'
    })
    token = login_response.json['token']

    # Тест обновления данных пользователя
    response = client.patch('/personal_account/update/', json={
        'first_name': 'Updated',
        'last_name': 'User',
        'email': 'test@example.com'
    }, headers={'Authorization': f'Bearer {token}'})
    assert response.status_code == 200
    assert response.json['message'] == 'User data updated successfully'

def test_logout(client, test_user):
    # Логин для получения токена
    login_response = client.post('/login/', json={
        'email': 'test@example.com',
        'password': 'secret'
    })
    token = login_response.json['token']

    # Тест логаута
    response = client.post('/personal_account/logout/', headers={'Authorization': f'Bearer {token}'})
    assert response.status_code == 200
    assert response.json['message'] == 'User logout correctly'

def test_delete_account(client, test_user):
    # Логин для получения токена
    login_response = client.post('/login/', json={
        'email': 'test@example.com',
        'password': 'secret'
    })
    token = login_response.json['token']

    # Тест удаления аккаунта
    response = client.delete('/personal_account/delete/', headers={'Authorization': f'Bearer {token}'})
    assert response.status_code == 200
    assert response.json['message'] == 'User was deleted successfully'
