#!/bin/sh
# Container entrypoint.
#
# Serving (`streamlit ...`, the image's default command) is gated, in this order:
#   1. required configuration is present (OPENAI_API_KEY);
#   2. the production index, if present, is writable by this user (see below);
#   3. the FULL release validation passes: validate-release-posture in its default mode, which
#      opens the index and checks its content against the canonical corpus (the static-only mode
#      can never exit 0 and is not used here).
# Only then does `exec` replace this shell with the server. Any failure exits non-zero before a
# server exists. Nothing here builds, repairs or creates an index: provisioning is a separate
# host-side operation (docs/07_index_provisioning.md), and validation prints the exact command.
#
# Every other command (validate-release-posture, answer-claim, sh) runs unchanged. Replacing the
# command is an explicit operator action; the application also re-validates the release posture
# before it builds a pipeline, so it never answers from an invalid index either way.
set -eu

INDEX_DIR="/app/data/04_index_production"

require_configuration() {
    # The value is never printed. A key with whitespace is a paste error, not a credential.
    case "${OPENAI_API_KEY:-}" in
        "" | *[[:space:]]*)
            echo "error: OPENAI_API_KEY is not set, is empty or contains whitespace in the container environment" >&2
            echo "start the stack with scripts/release_compose.py: it takes the key from your shell environment and never from a repository .env (docs/08_docker_runbook.md)" >&2
            exit 2
            ;;
    esac
}

require_writable_index() {
    # Chroma opens its SQLite database read-write even to answer queries: a read-only mount fails
    # inside the library with "attempt to write a readonly database". Say so before it does. A
    # missing or empty index is reported by the validation below with the build command.
    if [ -f "$INDEX_DIR/chroma.sqlite3" ] && { [ ! -w "$INDEX_DIR" ] || [ ! -w "$INDEX_DIR/chroma.sqlite3" ]; }; then
        echo "error: the production index at ${INDEX_DIR} is not writable by uid $(id -u)" >&2
        echo "Chroma needs write access to its own files even when it only reads: mount the index read-write and make the host directory writable by uid $(id -u) (docs/08_docker_runbook.md)" >&2
        exit 1
    fi
}

run_release_gate() {
    status=0
    validate-release-posture || status=$?
    if [ "$status" -ne 0 ]; then
        echo "error: release validation failed (exit ${status}); the server was not started" >&2
        exit "$status"
    fi
}

if [ "$#" -gt 0 ] && [ "$1" = "streamlit" ]; then
    require_configuration
    require_writable_index
    run_release_gate
fi

exec "$@"
