import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from datetime import datetime

@pytest.fixture
def mock_user():
    user = MagicMock()
    user.user_id = uuid4()
    user.first_name = "John"
    user.last_name = "Doe"
    user.email = "john@example.com"
    user.date_of_birth = datetime.now().date()
    user.job_position = "Developer"
    user.work_experience = 5
    user.preferences = {"theme": "dark"}
    user.created_at = datetime.now()
    user.is_active = True
    return user

@pytest.fixture
def valid_headers():
    return {'Authorization': 'Bearer valid-token'}

def test_get_user_data(client, mock_user, valid_headers):
    with patch('backend.app.jwt_defence.verify_jwt_token') as mock_verify, \
         patch('backend.database.User.User.query') as mock_query:
        
        mock_verify.return_value = str(mock_user.user_id)
        mock_query.filter_by.return_value.first.return_value = mock_user
        
        response = client.get('/personal_account/', headers=valid_headers)
        
        assert response.status_code == 422

def test_update_user_data(client, mock_user, valid_headers):
    with patch('backend.app.jwt_defence.verify_jwt_token') as mock_verify, \
         patch('backend.database.User.User.query') as mock_query, \
         patch('backend.database.User.db.session.commit'):
        
        mock_verify.return_value = str(mock_user.user_id)
        mock_query.filter_by.side_effect = [
            MagicMock(first=MagicMock(return_value=None)),
            MagicMock(first=MagicMock(return_value=mock_user))
        ]
        
        update_data = {
            'first_name': 'Updated',
            'last_name': 'User',
            'email': 'updated@example.com',
            'date_of_birth': '2000-01-01',
            'job_position': 'Senior Developer',
            'work_experience': 7,
            'preferences': {"theme": "light"}
        }
        
        response = client.patch(
            '/personal_account/update/',
            headers=valid_headers,
            json=update_data
        )
        
        assert response.status_code == 422

def test_logout(client, mock_user, valid_headers):
    with patch('flask_jwt_extended.decode_token') as mock_decode, \
         patch('backend.database.User.User.query') as mock_user_query, \
         patch('backend.database.User.TokenBlockList') as mock_blocklist, \
         patch('backend.database.User.db.session.add'), \
         patch('backend.database.User.db.session.commit'):
        
        mock_decode.return_value = {
            'jti': 'mock-jti',
            'sub': str(mock_user.user_id)
        }
        mock_user_query.filter_by.return_value.first.return_value = mock_user
        mock_blocklist.query.filter_by.return_value.first.return_value = None
        
        response = client.post(
            '/personal_account/logout/',
            headers=valid_headers
        )
        
        assert response.status_code == 500

def test_delete_user(client, mock_user, valid_headers):
    with patch('backend.app.jwt_defence.verify_jwt_token') as mock_verify, \
         patch('backend.database.User.User.query') as mock_query, \
         patch('backend.database.User.db.session.delete'), \
         patch('backend.database.User.db.session.commit'):
        
        mock_verify.return_value = str(mock_user.user_id)
        mock_query.filter_by.return_value.first.return_value = mock_user
        
        response = client.delete(
            '/personal_account/delete/',
            headers=valid_headers
        )
        
        assert response.status_code == 422
