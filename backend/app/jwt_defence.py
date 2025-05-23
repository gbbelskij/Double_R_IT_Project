from functools import wraps
from flask import request
from backend.app.jwt import verify_jwt_token
from flask_jwt_extended import decode_token
import redis

def token_required(f):
    @wraps(f)
    def decorated(self, *args, **kwargs):
        token = request.cookies.get('token')
        # Если токена нет — пробуем из заголовка (только для Swagger / тестов)
        if not token:
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]

        if not token:
            return {'message': 'Token is missing!'}, 401

        # Проверка токена (валиден ли и не истек ли он)
        user_id = verify_jwt_token(token)

        if not user_id:
            return {'message': 'Token is invalid or expired!'}, 401

        # Дополнительная проверка: токен заблокирован (например, после выхода из системы)
        # Используем decode_token для получения данных токена
        decoded_token = decode_token(token)  # Используем decode_token для получения данных токена
        jti = decoded_token['jti']  # Получаем идентификатор токена JTI
        
        # Проверяем, есть ли токен в черном списке
        r = redis.Redis(host='redis', port=6379, db=0)

        if r.exists(jti):
            return {"message": "The token has been revoked (logged out)."}, 401

        # Передаем user_id в функцию
        return f(self, user_id, decoded_token, jti, *args, **kwargs)

    return decorated