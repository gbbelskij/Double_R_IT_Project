#!/bin/bash
set -e

wait_for_db() {
    until pg_isready -h db -p 5432 -U myuser -d RRecomend_db; do
        echo "Waiting for database..."
        sleep 2
    done
}

wait_for_db
flask db upgrade
exec "$@"