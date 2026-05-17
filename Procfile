release: python manage.py migrate && python manage.py collectstatic --noinput
web: gunicorn config.wsgi:application --bind 0.0.0.0:${PORT:-8000} --timeout 120
worker: python worker.py
