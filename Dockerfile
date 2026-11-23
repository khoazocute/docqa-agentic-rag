FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    STORAGE_DIR=/app/storage

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY agents ./agents
COPY ui ./ui
RUN pip install --no-deps .

COPY data/demo ./data/demo
COPY eval/dataset.jsonl ./eval/dataset.jsonl

RUN useradd --create-home appuser && mkdir -p /app/storage && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "docqa.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
