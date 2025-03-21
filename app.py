from flask import Flask
from flask_migrate import Migrate
import config
from backend.database.User import db
from flask_restx import Api
from backend.handlers.register import register_ns
from backend.handlers.login import login_ns
from backend.handlers.personal_account import personal_account_ns
from flask_jwt_extended import JWTManager


def create_app():
    app = Flask(__name__)

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
    api.add_namespace(personal_account_ns, path='/personal_account')

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
