# Используем базовый образ Python
FROM python:3.10-slim

# Устанавливаем переменные окружения
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем исходный код приложения
COPY . .

# Открываем порт для доступа к приложению
EXPOSE 5000

# Запускаем приложение
CMD ["python3", "app.py"]
