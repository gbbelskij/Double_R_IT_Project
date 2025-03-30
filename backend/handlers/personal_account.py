from flask_restx import Namespace, Resource, fields
from flask import request
from backend.database.User import User, TokenBlockList, db
from backend.app.jwt_defence import token_required
from sqlalchemy.exc import IntegrityError
from flask_jwt_extended import decode_token
import redis
from datetime import datetime


personal_account_ns = Namespace('personal_account', description='User`s personal data')

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
    def get(self, user_id):
        user = User.query.filter_by(user_id=user_id).first()
        if user is None:
            return {'message': 'User not found'}, 404

        user_as_dict = {
            'user_id' : str(user.user_id),
            'first_name' : user.first_name,
            'last_name' : user.last_name,
            'email' : user.email,
            'date_of_birth' : str(user.date_of_birth),
            'job_position' : user.job_position,
            'work_experience' : user.work_experience,
            'preferences' : user.preferences,
            'created_at' : str(user.created_at),
        }
        
        return {'message': 'User got data correctly', 'user_data': user_as_dict}, 200


@personal_account_ns.route('/update/')
class UpdatePersonalAccountData(Resource):
    @personal_account_ns.expect(user_model)
    @personal_account_ns.response(200, "User data updated successfully")
    @personal_account_ns.response(404, "User not found")
    @personal_account_ns.response(500, "Internal server error")
    @token_required
    def patch(self, user_id):
        user = User.query.filter_by(user_id=user_id).first()
        if user is None:
            return {'message': 'User not found'}, 404

        data = request.get_json()

        first_name = data['first_name']
        last_name = data['last_name']
        email = data['email']
        date_of_birth = data.get('date_of_birth')
        job_position = data.get('job_position')
        work_experience = data.get('work_experience')
        preferences = data.get('preferences')

        # Проверка на существование пользователя с таким email
        if User.query.filter_by(email=email).first() and user.email != email:
            return {'message': 'Email already taken'}, 400

        user.first_name = first_name
        user.last_name = last_name
        user.email = email
        user.date_of_birth = date_of_birth
        user.job_position = job_position
        user.work_experience = work_experience
        user.preferences = preferences

        try:
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
    def post(self):
        token = None

        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split()[1]  # Получаем токен из заголовка Authorization

        if not token:
            return {'message': 'Token is missing!'}, 401
        try:
            decoded_token = decode_token(token)
            jti = decoded_token['jti']
            user_id = decoded_token['sub']
        except Exception as e:
            return {'message': f'Error: {str(e)}'}, 500
        
        user = User.query.filter_by(user_id=user_id).first()
        blocked_token = TokenBlockList(jti=jti, user_id=user_id)
        
        try:
            db.session.add(blocked_token)

            # добавление токена в редис
            r = redis.Redis(host='redis', port=6379, db=0)

            expiration_time = datetime.fromtimestamp(decoded_token['exp'])  # Преобразуем в datetime
            current_time = datetime.utcnow()

            # Вычисляем TTL (время жизни токена)
            ttl = (expiration_time - current_time).total_seconds()

            if ttl > 0:  # Только если TTL положительный (токен ещё действителен)
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
    def delete(self, user_id):
        try:
            user = User.query.filter_by(user_id=user_id).first()

            if not user:
                return {"message": "User not found"}, 404
            
            db.session.delete(user)
            db.session.commit()
            return {"message": "User was deleted successfully"}, 200
        except Exception as e:  
            return {"message": "Internal server error", "error": str(e)}, 500