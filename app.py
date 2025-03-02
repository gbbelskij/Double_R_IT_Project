from flask import Flask
from flask_migrate import Migrate
import config
from backend.database.User import db
from flask_restx import Api
from backend.handlers.register import register_ns

app = Flask(__name__)

# Пример URL для подключения
app.config['SQLALCHEMY_DATABASE_URI'] = config.DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
migrate = Migrate(app, db)

api = Api(app, version="1.0", title="RRecomend API",
          description="API сервиса RRecomend")
api.add_namespace(register_ns, path='/register')

def create_tables():
    db.create_all()


if __name__ == "__main__":
    with app.app_context():
        create_tables()
    app.run()
