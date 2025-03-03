from flask_restx import Namespace, Resource

test_ns = Namespace('test', description='User authentication operations')


@test_ns.route('/')
class LoginResource(Resource):
    from backend.app.jwt_defence import token_required
    @token_required
    @test_ns.doc(security='BearerAuth')  # Указываем на необходимость авторизации
    def post(self, user_id):
        # Ваш код для логина
        
        return {'message': f'Login successful {user_id}'}, 200