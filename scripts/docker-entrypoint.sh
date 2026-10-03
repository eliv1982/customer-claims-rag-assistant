#!/bin/sh
set -e

INDEX_DIR="/app/data/04_index_production"

run_production_preflight() {
    if [ ! -d "$INDEX_DIR" ]; then
        echo "error: production index mount missing at ${INDEX_DIR}" >&2
        echo "build it on the host from the repository (see docs/07_index_provisioning.md): validate-release-posture prints the exact command" >&2
        exit 1
    fi

    if [ -z "$(ls -A "$INDEX_DIR" 2>/dev/null || true)" ]; then
        echo "error: production index mount is empty at ${INDEX_DIR}" >&2
        echo "build it on the host from the repository (see docs/07_index_provisioning.md): validate-release-posture prints the exact command" >&2
        exit 1
    fi

    if [ ! -f "$INDEX_DIR/chroma.sqlite3" ]; then
        echo "error: chroma.sqlite3 not found in production index at ${INDEX_DIR}" >&2
        exit 1
    fi

    validate-release-posture
}

if [ "$#" -gt 0 ] && [ "$1" = "streamlit" ]; then
    run_production_preflight
fi

exec "$@"
