# Manual acceptance evidence

Committed artifacts supporting Stage 5C application-level acceptance for release `foodflow-10doc-release-v1`.

## Layout

| Path | Contents |
|------|----------|
| `terminal/` | Sanitized validator and fail-closed startup captures |
| `cli/` | Sanitized `answer-claim` JSON outputs |
| `ui/` | PNG screenshots from Docker Streamlit (Playwright) |

## Regeneration

1. Provision the active index (`docs/07_index_provisioning.md`).
2. Set `OPENAI_API_KEY` in the environment (do not commit).
3. `docker compose up -d`
4. `python scripts/capture_acceptance_cli.py`
5. `python scripts/capture_ui_screenshots.py` (requires `playwright` + Chromium)

See `evidence_manifest.json` for per-scenario checksums and metadata.
