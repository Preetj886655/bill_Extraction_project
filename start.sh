#!/usr/bin/env bash
exec gunicorn -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:${PORT:-8000} --workers 1
