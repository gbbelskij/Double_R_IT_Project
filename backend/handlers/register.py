from flask import request
from backend.database.User import User, db
from flask_restx import Resource, Namespace, fields
from backend.app.questions import questions

# Создание пространства имен для регистрации
register_ns = Namespace('register', description='User registration operations')

# Определение модели для Swagger документации
user_model = register_ns.model('User', {
    'first_name': fields.String(required=True, description='First name of the user'),
    'last_name': fields.String(required=True, description='Last name of the user'),
    'email': fields.String(required=True, description='Email of the user'),
    'password': fields.String(required=True, description='Password of the user'),
    'date_of_birth': fields.Date(required=False, description='Date of birth of the user'),
    'job_position': fields.String(description='Job position of the user'),
    'work_experience': fields.Integer(description='Work experience of the user'),
    'preferences': fields.Raw(description='Preferences of the user in JSON format')
})


@register_ns.route('/')
class Register(Resource):
    @register_ns.expect(user_model)
    @register_ns.response(201, 'User registered successfully')
    @register_ns.response(400, 'Missing required fields or email already taken')
    @register_ns.response(500, 'Internal server error')
    def post(self):
        """Register a new user"""
        data = request.get_json()

        if not data or not all(key in data for key in ['first_name', 'last_name', 'email', 'password']):
            return {'message': 'Missing required fields'}, 400

        first_name = data['first_name']
        last_name = data['last_name']
        email = data['email']
        password = data['password']
        date_of_birth = data.get('date_of_birth')
        job_position = data.get('job_position')
        work_experience = data.get('work_experience')
        preferences = data.get('preferences')

        # Проверка на существование пользователя с таким email
        if User.query.filter_by(email=email).first():
            return {'message': 'Email already taken'}, 400

        # Создаем нового пользователя
        new_user = User(
            first_name=first_name,
            last_name=last_name,
            email=email,
            date_of_birth=date_of_birth,
            job_position=job_position,
            work_experience=work_experience,
            preferences=preferences,
        )
        new_user.set_password(password)  # Хешируем пароль

        try:
            db.session.add(new_user)
            db.session.commit()
            return {'message': 'User registered successfully'}, 201
        except Exception as e:
            db.session.rollback()
            return {'message': f'Error: {str(e)}'}, 500


@register_ns.route('/questions/')
class Questions(Resource):
    @register_ns.response(200, 'Questions were sent correctly')
    def get(self):
        return {'questions': questions, 'message': 'Questions were sent correctly'}, 200
