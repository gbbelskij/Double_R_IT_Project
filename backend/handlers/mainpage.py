from flask_restx import Namespace, Resource
from backend.database.User import Course, User
from backend.app.jwt_defence import token_required


mainpage_ns = Namespace('mainpage', description='Courses information')

MOCK_COURSES = [
    {
        "id": 0,
        "title": "Python-разработчик",
        "duration": 10,
        "description": (
            "Вы освоите самый востребованный язык программирования, "
            "на котором пишут сайты, приложения, игры и чат-боты. "
            "Сделаете 3 проекта для портфолио, а Центр карьеры поможет найти работу."
        ),
        "url": "https://skillbox.ru/course/profession-python/",
        "image_url": "https://cdn.skillbox.pro/wbd-front/skillbox-static/main-page-new/mini-catalog/big/76-sm@1x.png"
    },
    {
        "id": 1,
        "title": "Бухгалтер",
        "duration": 6,
        "description": (
            "Вы научитесь вести бухучёт, работать в 1С, готовить налоговую отчётность "
            "и рассчитывать зарплату. Сможете начать карьеру или получить повышение."
        ),
        "url": "https://skillbox.ru/course/profession-accountant/",
        "image_url": "https://cdn.skillbox.pro/wbd-front/skillbox-static/main-page-new/mini-catalog/big/1008-sm@1x.png"
    },
    {
        "id": 2,
        "title": "Графический дизайнер",
        "duration": 12,
        "description": (
            "Вы научитесь создавать айдентику для брендов и освоите популярные графические редакторы – "
            "от Illustrator до Figma. Сможете зарабатывать уже во время обучения."
        ),
        "url": "https://skillbox.ru/course/profession-graphdesigner/",
        "image_url": "https://cdn.skillbox.pro/wbd-front/skillbox-static/main-page-new/mini-catalog/big/69-sm@1x.png"
    },
    {
        "id": 3,
        "title": "Python-разработчик",
        "duration": 10,
        "description": (
            "Вы освоите самый востребованный язык программирования, "
            "на котором пишут сайты, приложения, игры и чат-боты. "
            "Сделаете 3 проекта для портфолио, а Центр карьеры поможет найти работу."
        ),
        "url": "https://skillbox.ru/course/profession-python/",
        "image_url": "https://cdn.skillbox.pro/wbd-front/skillbox-static/main-page-new/mini-catalog/big/76-sm@1x.png"
    },
    {
        "id": 4,
        "title": "Бухгалтер",
        "duration": 6,
        "description": (
            "Вы научитесь вести бухучёт, работать в 1С, готовить налоговую отчётность "
            "и рассчитывать зарплату. Сможете начать карьеру или получить повышение."
        ),
        "url": "https://skillbox.ru/course/profession-accountant/",
        "image_url": "https://cdn.skillbox.pro/wbd-front/skillbox-static/main-page-new/mini-catalog/big/1008-sm@1x.png"
    },
    {
        "id": 5,
        "title": "Графический дизайнер",
        "duration": 12,
        "description": (
            "Вы научитесь создавать айдентику для брендов и освоите популярные графические редакторы – "
            "от Illustrator до Figma. Сможете зарабатывать уже во время обучения."
        ),
        "url": "https://skillbox.ru/course/profession-graphdesigner/",
        "image_url": "https://cdn.skillbox.pro/wbd-front/skillbox-static/main-page-new/mini-catalog/big/69-sm@1x.png"
    },
]


@mainpage_ns.route('/recommended_cources')
class RecommendedCourses(Resource):
    @mainpage_ns.response(404, 'No such user')
    @token_required
    def get(self, user_id, decoded_token, jti):
        user = User.query.filter_by(user_id=user_id).first()
        if user is None:
            return {'message': 'No such user'}, 404
        preferences = user.preferences

        '''ЗАГЛУШКА ПОЛУЧЕНИЕ РЕКОМЕНДОВАННЫХ КУРСОВ'''
        return {'message': 'courses', 'courses': MOCK_COURSES}, 200


@mainpage_ns.route('/all_cources')
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
                         'direction': course.direction,
                         'tags': course.tags}
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
