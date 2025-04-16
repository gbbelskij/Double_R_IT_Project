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

# Копируем и даем права на выполнение скрипта
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Открываем порт для доступа к приложению
EXPOSE 5000

# Запускаем скрипт, который сначала выполнит миграции, а затем запустит приложение
ENTRYPOINT ["bash", "entrypoint.sh"]
CMD ["python3", "app.py"]