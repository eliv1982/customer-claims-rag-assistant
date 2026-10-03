FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUSTOMER_CLAIMS_PROJECT_ROOT=/app

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install ".[ui]"

COPY configs ./configs
COPY prompts ./prompts
COPY scripts/docker-entrypoint.sh ./scripts/docker-entrypoint.sh

RUN sed -i 's/\r$//' ./scripts/docker-entrypoint.sh \
    && chmod +x ./scripts/docker-entrypoint.sh \
    && mkdir -p /app/data/04_index_production \
    && chown -R app:app /app

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)"

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["streamlit", "run", "src/customer_claims_rag/ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
