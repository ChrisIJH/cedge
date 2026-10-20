# syntax=docker/dockerfile:1

FROM python:3.9.11-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
COPY core/ ./core
COPY services/optimization/requirements.txt ./services/optimization/requirements.txt

RUN pip install --no-cache-dir --prefix=/install .
RUN pip install --no-cache-dir --prefix=/install -r services/optimization/requirements.txt


FROM python:3.9.11-slim AS runtime
WORKDIR /app

COPY --from=builder /install /usr/local
# The app is imported as a package (services.optimization.app) so it can use
# services/common.py — the same import path the tests and `python -m` use.
COPY services/__init__.py services/common.py ./services/
COPY services/optimization/__init__.py services/optimization/app.py ./services/optimization/

RUN useradd --create-home --shell /bin/false appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 2 services.optimization.app:app"]
