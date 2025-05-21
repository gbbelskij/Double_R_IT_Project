from flask_restx import Namespace, Resource, fields
from flask import request
from backend.database.User import User, TokenBlockList, db
from backend.app.jwt_defence import token_required
from sqlalchemy.exc import IntegrityError
from flask_jwt_extended import decode_token
import redis
from datetime import datetime


personal_account_ns = Namespace(
    'personal_account', description='User`s personal data')

# Определение модели для Swagger документации
user_model = personal_account_ns.model('User_data', {
    'first_name': fields.String(description='First name of the user'),
    'last_name': fields.String(description='Last name of the user'),
    'email': fields.String(description='Email of the user'),
    'date_of_birth': fields.Date(description='Date of birth of the user'),
    'job_position': fields.String(description='Job position of the user'),
    'work_experience': fields.Integer(description='Work experience of the user'),
    'preferences': fields.Raw(description='Preferences of the user in JSON format')
})


@personal_account_ns.route('/')
class PersonalAccount(Resource):
    @personal_account_ns.response(200, 'User got data correctly')
    @personal_account_ns.response(400, 'Invalid credentials')
    @token_required
    def get(self, user_id, decoded_token, jti):
        try:
            user = User.query.filter_by(user_id=user_id).first()

            if user is None:
                return {'message': 'User not found'}, 404

            user_as_dict = {
                'user_id': str(user.user_id),
                'first_name': user.first_name,
                'last_name': user.last_name,
                'email': user.email,
                'date_of_birth': str(user.date_of_birth),
                'job_position': user.job_position,
                'work_experience': user.work_experience,
                'preferences': user.preferences,
                'created_at': str(user.created_at),
            }

            return {'message': 'User got data correctly', 'user_data': user_as_dict}, 200

        except Exception as e:
            return {'message': f'Error: {str(e)}'}, 500


@personal_account_ns.route('/update/')
class UpdatePersonalAccountData(Resource):
    @personal_account_ns.expect(user_model)
    @personal_account_ns.response(200, "User data updated successfully")
    @personal_account_ns.response(400, "Bad request")
    @personal_account_ns.response(404, "User not found")
    @personal_account_ns.response(500, "Internal server error")
    @token_required
    def patch(self, user_id, decoded_token, jti):
        try:
            user = User.query.filter_by(user_id=user_id).first()

            if user is None:
                return {'message': 'User not found'}, 404

            data = request.get_json()

            if 'first_name' in data:
                user.first_name = data['first_name']

            if 'last_name' in data:
                user.last_name = data['last_name']

            if 'email' in data:
                email = data['email']
                if User.query.filter(User.email == email, User.user_id != user_id).first():
                    return {'message': 'Email already taken'}, 400
                user.email = email

            if 'date_of_birth' in data:
                user.date_of_birth = data['date_of_birth']

            if 'job_position' in data:
                user.job_position = data['job_position']

            if 'work_experience' in data:
                user.work_experience = data['work_experience']

            if 'preferences' in data:
                user.preferences = data['preferences']

            if 'password' in data and data['password'].strip():
                if 'old_password' not in data:
                    return {'message': 'Old password is required to set a new password'}, 400

                if not user.check_password(data['old_password']):
                    return {'message': 'Old password is incorrect'}, 400

                user.password_hash = user.set_password(data['password'])

            user.updated_at = db.func.current_timestamp()
            db.session.commit()

            return {'message': 'User data updated successfully'}, 200

        except Exception as e:
            db.session.rollback()
            return {'message': f'Error: {str(e)}'}, 500


@personal_account_ns.route('/logout/')
class Logout(Resource):
    @personal_account_ns.response(200, 'User logout correctly')
    @personal_account_ns.response(400, 'Invalid token')
    @personal_account_ns.response(401, 'Token is missing!')
    @token_required
    def post(self, user_id, decoded_token, jti):
        try:
            user = User.query.filter_by(user_id=user_id).first()
            blocked_token = TokenBlockList(jti=jti, user_id=user_id)

            db.session.add(blocked_token)

            # добавление токена в редис
            r = redis.Redis(host='redis', port=6379, db=0)

            expiration_time = datetime.fromtimestamp(
                decoded_token['exp'])  # Преобразуем в datetime
            current_time = datetime.utcnow()

            # Вычисляем TTL (время жизни токена)
            ttl = (expiration_time - current_time).total_seconds()

            # Только если TTL положительный (токен ещё действителен)
            if ttl > 0:
                r.setex(jti, int(ttl), 'blacklisted')

            user.last_login = db.func.current_timestamp()
            user.is_active = False
            db.session.commit()

            return {'message': 'User logout correctly'}

        except IntegrityError:
            db.session.rollback()

            return {'message': 'User has already logged out'}, 500

        except Exception as e:
            db.session.rollback()

            return {'message': f'Error: {str(e)}'}, 500


@personal_account_ns.route('/delete/')
class Delete(Resource):
    @personal_account_ns.response(200, "User was deleted successfully")
    @personal_account_ns.response(404, "User not found")
    @personal_account_ns.response(500, "Internal server error")
    @token_required
    def delete(self, user_id, decoded_token, jti):
        try:
            user = User.query.filter_by(user_id=user_id).first()

            if not user:
                return {"message": "User not found"}, 404

            db.session.delete(user)
            db.session.commit()

            return {"message": "User was deleted successfully"}, 200

        except Exception as e:
            db.session.rollback()

            return {"message": "Internal server error", "error": str(e)}, 500
