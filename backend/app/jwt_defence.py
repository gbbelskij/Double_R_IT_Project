from functools import wraps
from flask import request
from backend.app.jwt import verify_jwt_token

def token_required(f):
    @wraps(f)
    def decorated(self, *args, **kwargs):
        token = None

        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split()[1]  # Получаем токен из заголовка Authorization

        if not token:
            return {'message': 'Token is missing!'}, 401

        user_id = verify_jwt_token(token)
        
        if not user_id:
            return {'message': 'Token is invalid or expired!'}, 401

        return f(self, user_id, *args, **kwargs)
    return decorated
