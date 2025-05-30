from flask_restx import Namespace, Resource, fields
from flask import request
from backend.database.User import User, db


login_ns = Namespace('login', description='User authentication operations')

login_model = login_ns.model('Login', {
    'email': fields.String(required=True, description='Email of the user'),
    'password': fields.String(required=True, description='Password of the user')
})


@login_ns.route('/')
class Login(Resource):
    @login_ns.expect(login_model)
    @login_ns.response(200, 'Login successful')
    @login_ns.response(400, 'Invalid credentials')
    def post(self):
        """Login and get JWT"""
        from backend.app.jwt import create_jwt_token

        data = request.get_json()
        email = data['email']
        password = data['password']

        user = User.query.filter_by(email=email).first()

        if not user:
            return {'message': 'User not found'}, 400

        if not user.check_password(password):
            return {'message': 'Invalid password'}, 400

        token = create_jwt_token(user.user_id)
        user.is_active = True
        db.session.commit()

        return {'message': 'Login successful', 'token': token}, 200
