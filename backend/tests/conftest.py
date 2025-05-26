import pytest
from unittest.mock import MagicMock, patch
from app import create_app

@pytest.fixture(scope='module')
def app():
    app = create_app()
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'JWT_SECRET_KEY': 'test-secret-key',
        'JWT_ALGORITHM': 'HS256'
    })
    
    # Мокируем JWT и базу данных
    with patch('backend.app.jwt.verify_jwt_token', return_value='user123'), \
         patch('backend.database.User.db.session'):
        yield app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def auth_headers():
    return {'Authorization': 'Bearer valid-token'}
