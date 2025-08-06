web: gunicorn --bind 0.0.0.0:$PORT --timeout 180 --workers 1 --worker-class sync --max-requests 50 --max-requests-jitter 10 --preload wsgi:app
