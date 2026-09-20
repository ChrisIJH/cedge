# syntax=docker/dockerfile:1

FROM python:3.9.11-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
COPY core/ ./core
COPY services/marketdata/requirements.txt ./services/marketdata/requirements.txt

RUN pip install --no-cache-dir --prefix=/install .
RUN pip install --no-cache-dir --prefix=/install -r services/marketdata/requirements.txt


FROM python:3.9.11-slim AS runtime
WORKDIR /app

COPY --from=builder /install /usr/local
COPY services/marketdata/app.py ./app.py

RUN useradd --create-home --shell /bin/false appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 2 app:app"]