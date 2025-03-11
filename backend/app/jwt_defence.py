from functools import wraps
from flask import request
from backend.app.jwt import verify_jwt_token
from flask_jwt_extended import decode_token
from backend.database.User import TokenBlockList


def token_required(f):
    @wraps(f)
    def decorated(self, *args, **kwargs):
        token = None

        # Извлечение токена из заголовков запроса
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split()[1]  # Получаем токен из заголовка Authorization

        if not token:
            return {'message': 'Token is missing!'}, 401

        # Проверка токена (валиден ли и не истек ли он)
        user_id = verify_jwt_token(token)

        if not user_id:
            return {'message': 'Token is invalid or expired!'}, 401

        # Дополнительная проверка: токен заблокирован (например, после выхода из системы)
        decoded_token = decode_token(token)  # Используем decode_token для получения данных токена
        jti = decoded_token['jti']  # Получаем идентификатор токена JTI
        
        # Проверяем, есть ли токен в черном списке
        token_in_blocklist = TokenBlockList.query.filter_by(jti=jti).first()

        if token_in_blocklist:
            return {"message": "The token has been revoked (logged out)."}, 401

        # Передаем user_id в функцию
        return f(self, user_id, *args, **kwargs)

    return decorated
