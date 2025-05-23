import os

DATABASE_URL = os.environ.get(
    "DATABASE_URL", 
    "postgresql+psycopg2://myuser:mypassword@db:5432/RRecomend_db"
)
SECRET_KEY = os.environ.get("SECRET_KEY", 'MAI_IS_THE_BEST_PLACE_FOR_VIPERRS')
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", 'HS256')