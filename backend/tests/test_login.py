import pytest
from unittest.mock import patch, MagicMock

from unittest.mock import patch, MagicMock

def test_register_success(client):
    with patch('backend.handlers.register.User') as mock_user_model:
        mock_user_model.query.filter_by.return_value.first.return_value = None
        response = client.post('/register/', json={
            'first_name': 'John',
            'last_name': 'Doe',
            'email': 'john@example.com',
            'password': 'password123'
        })
        assert response.status_code == 201

def test_login_success(client):
    mock_user = MagicMock()
    mock_user.check_password.return_value = True
    mock_user.user_id = 'user123'
    
    with patch('backend.handlers.login.User') as mock_user_model:
        mock_user_model.query.filter_by.return_value.first.return_value = mock_user
        response = client.post('/login/', json={
            'email': 'test@example.com',
            'password': 'correctpass'
        })
        assert response.status_code == 200
        assert 'token' in response.json

def test_login_invalid_password(client):
    mock_user = MagicMock()
    mock_user.check_password.return_value = False
    
    with patch('backend.handlers.login.User') as mock_user_model:
        mock_user_model.query.filter_by.return_value.first.return_value = mock_user
        response = client.post('/login/', json={
            'email': 'test@example.com',
            'password': 'wrongpass'
        })
        assert response.status_code == 400
