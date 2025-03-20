#!/bin/bash
# Выполняем миграции базы данных
flask db upgrade

# Запускаем приложение
exec "$@"
