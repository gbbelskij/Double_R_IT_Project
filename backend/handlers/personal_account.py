from flask_restx import Namespace, Resource
from flask import request
from backend.database.User import User, db
from backend.app.jwt_defence import token_required


personal_account_ns = Namespace('personal_account', description='User`s personal data')


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