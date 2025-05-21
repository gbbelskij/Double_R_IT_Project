import time
from sqlalchemy.exc import OperationalError
from sqlalchemy import text

def wait_for_db(db, timeout=30):
    start_time = time.time()
    while True:
        try:
            # Попытка выполнить простой запрос
            db.session.execute(text('SELECT 1'))
            return
        except OperationalError:
            if time.time() - start_time > timeout:
                raise TimeoutError("Database connection timed out.")
            print("⏳ Waiting for database to be ready...")
            time.sleep(1)
