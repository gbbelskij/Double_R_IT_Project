from flask_restx import Namespace, Resource
from backend.database.User import Course, User
from backend.app.jwt_defence import token_required
from backend.ml_model.model.course_rec import model
import uuid


mainpage_ns = Namespace('mainpage', description='Courses information')


@mainpage_ns.route('/recommended_cources/')
class RecommendedCourses(Resource):
    @mainpage_ns.response(404, 'No such user')
    @token_required
    def get(self, user_id, decoded_token, jti):
        from app import app
        with app.app_context():
            courses_ids = model(app, user_id)  # список UUID или строк-UUID
            courses = []
            for one_course in courses_ids:
                try:
                    course_uuid = uuid.UUID(str(one_course))
                    course = Course.query.filter_by(course_id=course_uuid).first()
                    if course:
                        courses.append(course)
                except Exception as e:
                    print(f"[ERROR] Не удалось обработать course_id {one_course!r}: {e}")

            # Формируем итоговый список для API
            courses_list = [
                {
                    'course_id': str(course.course_id),
                    'title': course.title,
                    'duration': course.duration,
                    'url': course.link,
                    'description': course.description,
                    'price': course.price,
                    'type': course.type,
                    'direction': course.direction
                }
                for course in courses
            ]

        return {'message': 'courses', 'courses': courses_list}, 200

@mainpage_ns.route('/all_cources/')
class PersonalAccount(Resource):
    @mainpage_ns.response(200, 'Courses data got correctly')
    @token_required
    def get(self, user_id, decoded_token, jti):
        courses = Course.query.all()

        courses_list = [{'course_id': str(course.course_id),
                         'title': course.title,
                         'duration': course.duration,
                         'url': course.link,
                         'description': course.description,
                         'price': course.price,
                         'type': course.type,
                         'direction': course.direction}
                        for course in courses]

        return {'message': 'Courses data got correctly', 'courses': courses_list}, 200


@mainpage_ns.route('/<uuid:course_id>/')
class PersonalAccount(Resource):
    @mainpage_ns.response(200, 'Course data got correctly')
    @mainpage_ns.response(404, 'Course not found')
    @token_required
    def get(self, user_id, decoded_token, jti, course_id):
        course = Course.query.filter_by(course_id=course_id).first()

        if course is None:
            return {'message': 'Course not found'}, 404

        course_as_dict = {
            'course_id': str(course.course_id),
            'title': course.title,
            'link': course.link,
            'duration': course.duration,
            'description': course.description,
            'price': course.price,
            'type': course.type,
            'direction': course.direction
        }

        return {'message': 'Course data got correctly', 'data': course_as_dict}, 200
