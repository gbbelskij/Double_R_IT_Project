from flask import Flask
from flask_cors import CORS
from flask_migrate import Migrate
import config
from backend.database.User import db, Course
from flask_restx import Api
from backend.handlers.register import register_ns
from backend.handlers.login import login_ns
from backend.handlers.verify import verify_ns
from backend.handlers.personal_account import personal_account_ns
from backend.handlers.mainpage import mainpage_ns
import csv
import uuid
from backend.app.wait_bd import wait_for_db
import json


from flask_jwt_extended import JWTManager


def create_app():
    app = Flask(__name__)
    


    CORS(app, supports_credentials=True, resources={r"/*": {"origins": [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4173",
        "http://localhost:5173",
    ]}})

    # Пример URL для подключения
    app.config['SQLALCHEMY_DATABASE_URI'] = config.DATABASE_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['JWT_ALGORITHM'] = config.JWT_ALGORITHM

    db.init_app(app)
    migrate = Migrate(app, db)
    api = Api(
        app,
        version="1.0",
        title="RRecomend API",
        description="API сервиса RRecomend",
        security=[{'BearerAuth': []}],
        authorizations={
            'BearerAuth': {
                'type': 'apiKey',
                'in': 'header',
                'name': 'Authorization'
            }
        }
    )

    jwt = JWTManager(app)

    api.add_namespace(register_ns, path='/register')
    api.add_namespace(login_ns, path='/login')
    api.add_namespace(verify_ns, path='/verify')
    api.add_namespace(personal_account_ns, path='/personal_account')
    api.add_namespace(mainpage_ns, path='/mainpage')

    return app


app = create_app()
if __name__ == "__main__":

    with app.app_context():

        wait_for_db(db)

        with open("courses.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tags = json.loads(row["tags"].replace("'", '"'))
                existing = db.session.query(Course).filter_by(description=row["description"]).first()
                if not existing:
                    course = Course(
                        course_id=uuid.uuid4(),
                        title=row["title"],
                        link=row["link"],
                        duration=row["duration"],
                        description=row["description"],
                        price=row["price"],
                        type=row["type"],
                        direction=row["direction"],
                        tags=tags,
                    )
                    db.session.add(course)
            db.session.commit()
    app.run(host="0.0.0.0", port=5000)