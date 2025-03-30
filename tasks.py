from celery import Celery
import redis
from backend.database.User import db, TokenBlockList
from app import create_app  # Импортируйте функцию для создания приложения

# Настройка Celery для периодической задачи
celery_app = Celery('tasks', broker='redis://redis_db:6379/0')

# Подключение к Redis
r = redis.Redis(host='redis_db', port=6379, db=0)

# Создайте приложение Flask
flask_app = create_app()

@celery_app.task
def cleanup_blacklist():
    # Запускаем контекст приложения Flask
    with flask_app.app_context():
        # Получаем все токены из черного списка PostgreSQL
        blacklisted_tokens = db.session.query(TokenBlockList).all()
        for token in blacklisted_tokens:
            token_jti = token.jti
            # Проверяем, есть ли токен в Redis
            if not r.exists(token_jti):
                # Если токена нет в Redis, удаляем его из PostgreSQL
                db.session.delete(token)
        db.session.commit()  # Не забудьте зафиксировать изменения

# Настройка периодического выполнения задачи (например, раз в день)
celery_app.conf.beat_schedule = {
    'cleanup-blacklist-every-day': {
        'task': 'tasks.cleanup_blacklist',
        'schedule': 86400.0,  # Выполнять раз в сутки (86400 секунд)
    },
}
