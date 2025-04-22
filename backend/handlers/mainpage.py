from flask_restx import Namespace, Resource
from backend.database.User import Course, User
from backend.app.jwt_defence import token_required


mainpage_ns = Namespace('mainpage', description='Courses information')


@mainpage_ns.route('/')
class RecommendedCourses(Resource):
    @mainpage_ns.response(404, 'No such user')
    @token_required
    def get(self, user_id):
        user = User.query.filter_by(user_id=user_id).first()
        if user is None:
            return {'message': 'No such user'}, 404
        preferences = user.preferences

        '''ЗАГЛУШКА ПОЛУЧЕНИЕ РЕКОМЕНДОВАННЫХ КУРСОВ'''
        return {'message': 'courses'}


@mainpage_ns.route('/all')
class PersonalAccount(Resource):
    @mainpage_ns.response(200, 'Courses data got correctly')
    @token_required
    def get(self, user_id):
        courses = Course.query.all()

        courses_list = [{'course_id' : str(course.course_id),
                         'title': course.title,
                         'link': course.link,
                         'duration': course.duration,
                         'description': course.description,
                         'price': course.price,
                         'type': course.type,
                         'direction': course.direction} 
                         for course in courses]

        return {'message': 'Courses data got correctly', 'courses_data': courses_list}, 200


@mainpage_ns.route('/<uuid:course_id>/')
class PersonalAccount(Resource):
    @mainpage_ns.response(200, 'Course data got correctly')
    @mainpage_ns.response(404, 'Course not found')
    @token_required
    def get(self, user_id, course_id):
        course = Course.query.filter_by(course_id=course_id).first()

        if course is None:
            return {'message': 'Course not found'}, 404
        
        course_as_dict = {
            'course_id' : str(course.course_id),
            'title': course.title,
            'link': course.link,
            'duration': course.duration,
            'description': course.description,
            'price': course.price,
            'type': course.type,
            'direction': course.direction
        }

        return {'message': 'Course data got correctly', 'course_data': course_as_dict}, 200

