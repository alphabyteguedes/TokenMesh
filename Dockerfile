FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml README.md ./
COPY core/ core/
COPY web/ web/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=builder /app/core ./core
COPY --from=builder /app/web ./web

RUN mkdir -p /app/data && \
    useradd --create-home --shell /bin/bash tokenmesh && \
    chown -R tokenmesh:tokenmesh /app

USER tokenmesh

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8787/healthz')" || exit 1

ENTRYPOINT ["uvicorn", "core.main:app", "--host", "0.0.0.0", "--port", "8787"]
