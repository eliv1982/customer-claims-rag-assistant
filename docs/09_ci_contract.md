# CI contract

`.github/workflows/ci.yml` checks, on every push to `main` and every pull request, that a normal GitHub checkout installs from its declared dependencies and passes its tests with nothing but the repository: no credential, no `.env`, no built index, no tokenizer download and no application network access. It deploys nothing, uploads nothing and has read-only repository permissions.

## What runs where

| Job | Runner | Tokenizer | Command |
|---|---|---|---|
| `default-suite` | Ubuntu and Windows, Python 3.12 | offline stand-in (**not** `cl100k_base`) | `python -m pytest -p no:cacheprovider` |
| `real-tokenizer` | Ubuntu, Python 3.12 | real `cl100k_base` | `python -m pytest --real-tiktoken -m real_tiktoken -p no:cacheprovider` |

Both jobs install with `python -m pip install -e ".[dev]"` and run `python -m pip check`. Nothing else is installed (no Playwright/`screenshots` extra); a dependency the suite needs but `pyproject.toml` does not declare fails the job, and is fixed in `pyproject.toml`.

Why two lanes: the default suite replaces the tokenizer with a deterministic word-level stand-in so it needs no vocabulary and no network. That stand-in packs chunks differently from production, so the default lane cannot say anything about production chunk counts, corpus fingerprints or document topology. The `real_tiktoken` tests (about the canonical corpus topology and golden fingerprints) skip there, and the real-tokenizer job is what runs them. A run of the real lane never skips: a `real_tiktoken` test that skips there is reported as a failure.

The default suite runs on both Ubuntu and Windows because the repository's path, line-ending and symlink handling is platform-sensitive; the real lane is a function of the corpus and the vocabulary only, so one platform is enough.

## Network boundary

| Phase | Network |
|---|---|
| package installation | allowed |
| tokenizer provisioning (`scripts/provision_tiktoken_cache.py`, its own step) | allowed, to the tiktoken vocabulary host only |
| test execution | no application or tokenizer access |

During pytest the suite's own network guard (`tests/network_guard.py`) raises on any non-loopback DNS lookup or connect and fails the session even if the code under test swallowed the error. The test steps additionally set `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` to a dead loopback address, a second barrier that also covers subprocesses the tests start. This is not an operating-system sandbox: the runner itself still has internet access.

## Real `cl100k_base` provisioning

The vocabulary is not committed (its redistribution terms are unresolved). The `real-tokenizer` job:

1. restores an optional GitHub Actions cache of `TIKTOKEN_CACHE_DIR` (`/tmp/ci-tiktoken-cache`);
2. always runs `python scripts/provision_tiktoken_cache.py`, which requires an absolute `TIKTOKEN_CACHE_DIR`, loads `cl100k_base` through tiktoken (a cache hit needs no network; a missing, truncated or tampered file fails tiktoken's built-in SHA-256 check and is fetched again), verifies name, vocabulary size and reference tokens, and prints the cache contents with their digests;
3. runs a negative control: the lane against an empty cache must stop at setup with pytest usage error 4, proving it cannot pass, skip or fall back to the stand-in without the vocabulary;
4. makes the cache read-only and runs the lane, which reads the provisioned cache and refuses to download (`tests/tokenizer_lanes.py`).

The Actions cache is only an optimisation: a miss provisions, and a stale or corrupted entry cannot satisfy the lane because step 2 and the lane both verify the file.

## What CI deliberately does not do

- It never calls OpenAI and never needs `OPENAI_API_KEY`; the workflow references no secret.
- It does not build the production Chroma index or run retrieval evaluation. The index is built on purpose, with a key in the process environment, as described in `docs/07_index_provisioning.md`; the canonical corpus identity (documents, chunk topology, fingerprint) and the release validator's store-recomputation logic are what CI verifies.
- It does not run the `local_artifact` tests, which verify gitignored local artifacts (historical archives, a built production index). They skip with the missing path as reason; the repository-controlled parts of those contracts are covered by ordinary tests.
- It does not build or scan the Docker image.

## Reproducing the lanes locally

```bash
python -m pip install -e ".[dev]" && python -m pip check
python -m pytest                                   # default lane
TIKTOKEN_CACHE_DIR=/abs/path python scripts/provision_tiktoken_cache.py
TIKTOKEN_CACHE_DIR=/abs/path python -m pytest --real-tiktoken -m real_tiktoken
```

Use a virtual environment outside the checkout; keep the same `TIKTOKEN_CACHE_DIR` for provisioning and for the lane.
