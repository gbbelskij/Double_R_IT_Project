from flask_restx import Namespace, Resource
from flask import request
from backend.database.User import User
from backend.app.jwt import verify_jwt_token

verify_ns = Namespace('verify', description='JWT token verification')


@verify_ns.route('/')
class Verify(Resource):
    @verify_ns.response(200, 'Token is valid')
    @verify_ns.response(401, 'Invalid or missing token')
    @verify_ns.response(404, 'User not found')
    def get(self):
        """Verify JWT token validity"""
        token = request.cookies.get('token')

        if not token:
            return {'message': 'Missing or invalid token'}, 401

        try:
            user_id = verify_jwt_token(token)

            user = User.query.filter_by(user_id=user_id).first()

            if not user:
                return {'message': 'User not found'}, 404

            return {'message': 'Token is valid'}, 200

        except Exception as e:
            return {'message': str(e)}, 401
