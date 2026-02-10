FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml ./
COPY README.md ./
COPY semantic_core ./semantic_core
COPY semantic_api ./semantic_api

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "semantic_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


