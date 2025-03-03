import jwt
from datetime import datetime, timedelta
from app import app

def create_jwt_token(user_id):
    """Создаем JWT токен для пользователя"""
    payload = {
        'user_id': str(user_id),
        'exp': datetime.utcnow() + timedelta(hours=24)  # Токен действует 24 часа
    }
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm=app.config['JWT_ALGORITHM'])
    return token

def verify_jwt_token(token):
    """Проверяем JWT токен"""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=[app.config['JWT_ALGORITHM']])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None  # Токен истек
    except jwt.InvalidTokenError:
        return None  # Неверный токен