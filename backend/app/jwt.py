from datetime import timedelta
from flask_jwt_extended import (
    create_access_token, decode_token
)


def create_jwt_token(user_id):
    """Создаем JWT токен для пользователя с помощью Flask-JWT-Extended"""
    # Создаем токен доступа (access token), срок действия 24 часа
    access_token = create_access_token(
        identity=user_id, expires_delta=timedelta(hours=24))
    return access_token


def verify_jwt_token(token):
    """Проверяем JWT токен с помощью Flask-JWT-Extended"""
    try:
        decoded_token = decode_token(token)
        return decoded_token['sub']  # 'sub' содержит user_id
    except Exception as e:
        print(f"Error verifying token: {str(e)}")
        return None
