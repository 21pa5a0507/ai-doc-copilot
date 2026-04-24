FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY backend/requirements-serving.txt ./backend/requirements-serving.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r backend/requirements-serving.txt

COPY backend/ ./backend/
COPY storage/ ./storage/

EXPOSE 8000

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
