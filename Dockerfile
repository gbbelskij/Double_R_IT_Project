# Используем базовый образ Python
FROM python:3.10-slim

# Устанавливаем postgresql-client для pg_isready
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

# Устанавливаем переменные окружения
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем исходный код приложения
COPY . .
COPY migrations/ ./migrations/

# Копируем и даем права на выполнение скрипта
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Открываем порт для доступа к приложению
EXPOSE 5000

# Запускаем скрипт, который сначала выполнит миграции, а затем запустит приложение
ENTRYPOINT ["bash", "entrypoint.sh"]
CMD ["python3", "app.py"]