"""The locally built production index against the committed release posture.

Everything about the production release that can be checked from the repository is checked by
default tests (canonical corpus manifest, descriptor, validator mutations on staged indexes).
What remains is the one thing the repository cannot contain: the Chroma index itself, built with
live OpenAI embeddings by

    python -m customer_claims_rag.cli.build_index --corpus-manifest \\
        configs/corpus/foodflow_production_v1.json --index-dir data/04_index_production ... --rebuild

This ``local_artifact`` test skips (naming the path) while that index has not been built and
fails if it exists but does not satisfy the posture: stale or hand-edited indexes must be seen.
"""

from __future__ import annotations

import json
from pathlib import Path

from customer_claims_rag.release.posture import resolve_production_release_posture
from customer_claims_rag.release.readiness import assess_release_readiness
from tests.local_artifacts import requires_local_artifacts

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTOR = json.loads((ROOT / "configs" / "release" / "production_posture.json").read_text(encoding="utf-8"))
PRODUCTION_INDEX = ROOT / DESCRIPTOR["targets"][DESCRIPTOR["default_target"]]["index_path"]


@requires_local_artifacts(
    PRODUCTION_INDEX / "manifest.json",
    why="the production index built from the canonical corpus with live OpenAI embeddings",
)
def test_local_production_index_satisfies_the_release_posture() -> None:
    readiness = assess_release_readiness(resolve_production_release_posture(project_root=ROOT))
    assert readiness.release_can_proceed, readiness.problem
    assert readiness.index_matches_canonical_corpus == "yes"
    assert readiness.diagnostics is not None
    assert readiness.diagnostics.integrity == "store_recomputed"
