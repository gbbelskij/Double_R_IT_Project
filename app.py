from flask import Flask
import config
from backend.database.User import User, db

app = Flask(__name__)

# Пример URL для подключения
app.config['SQLALCHEMY_DATABASE_URI'] = config.DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def create_tables():
    db.create_all()
    print("Таблицы созданы!")

if __name__ == "__main__":
    with app.app_context():
        create_tables()
    app.run()
