from flask_restx import Namespace, Resource
from flask import request
from backend.database.User import User, db, Interaction
from backend.app.jwt_defence import token_required

interaction_ns = Namespace('login', description='User authentication operations')


@interaction_ns.route('/')
class Interact(Resource):
    @interaction_ns.response(200, 'Interaction was seted successfully')
    @token_required
    def post(self, user_id, decoded_token, jti):
        """Set Interaction of user"""

        data = request.get_json()
        course_id = data['course_id']

        new_interaction = Interaction(
            user_id = user_id,
            course_id = course_id,
            liked = True
        )

        try:
            db.session.add(new_interaction)
            db.session.commit()
            return {'message': 'Interaction was seted successfully'}, 200
        except Exception as e:
            db.session.rollback()
            return {'message': f'Error: {str(e)}'}, 500