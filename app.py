from flask import Flask
from flask_migrate import Migrate
import config
from backend.database.User import db
from flask_restx import Api
from backend.handlers.register import register_ns
from backend.handlers.login import login_ns

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


api.add_namespace(register_ns, path='/register')
api.add_namespace(login_ns, path='/login')

def create_tables():
    db.create_all()


if __name__ == "__main__":
    with app.app_context():
        create_tables()
    app.run()
