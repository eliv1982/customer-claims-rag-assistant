# Runtime image for the Streamlit UI: the application and its `ui` extra only. No dev/test/CI
# tooling, no tokenizer vocabulary (it is needed to build an index, not to serve one), no
# credential and no index: canonical source Markdown is included so the release gate can recompute
# source identity, while the production index is bind-mounted at run time and never baked in.
# Nothing in this file needs a secret; do not pass one with --build-arg.
FROM python:3.12-slim

# No telemetry and no phoning home from the server: Chroma and Streamlit usage statistics are off,
# and a set browser address stops headless Streamlit from asking a third-party service for this
# host's public IP at every start (to print an "External URL"). Binding stays inside the container;
# which host interface publishes the port is decided by compose.yaml.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUSTOMER_CLAIMS_PROJECT_ROOT=/app \
    ANONYMIZED_TELEMETRY=False \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_BROWSER_SERVER_ADDRESS=localhost \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

# Dedicated runtime user, numeric so that `runAsNonRoot`-style checks can verify it. It owns
# nothing under /app: the application tree stays root-owned and is not writable by the process.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin app

COPY pyproject.toml README.md ./
COPY src ./src

# `rm` removes the in-tree build leftovers that `pip install .` creates next to the sources.
RUN pip install ".[ui]" \
    && rm -rf build src/*.egg-info

COPY configs ./configs
COPY prompts ./prompts
COPY data/02_clean_markdown ./data/02_clean_markdown
COPY scripts/docker-entrypoint.sh scripts/healthcheck.py ./scripts/

# The mount point only: it stays empty and root-owned, and no step of this image fills it.
RUN sed -i 's/\r$//' scripts/docker-entrypoint.sh scripts/healthcheck.py \
    && chmod 0555 scripts/docker-entrypoint.sh scripts/healthcheck.py \
    && mkdir -p data/04_index_production

USER 10001:10001

EXPOSE 8501

# Local liveness only. It does not replace the release gate in the entrypoint: an invalid or missing
# index never gets as far as a server that could report healthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "/app/scripts/healthcheck.py"]

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["streamlit", "run", "src/customer_claims_rag/ui/streamlit_app.py"]
