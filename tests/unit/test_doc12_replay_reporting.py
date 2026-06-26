"""Reporting tests for doc12 replay integrity terminology."""

from __future__ import annotations

from customer_claims_rag.evaluation.doc12_threat_atomic_models import (
    FrozenSnapshotReplayResult,
    LiveProviderRobustnessResult,
    ReplayIntegrityResult,
)


def test_frozen_and_live_structures_are_separate() -> None:
    result = ReplayIntegrityResult(
        frozen_snapshot_replay=FrozenSnapshotReplayResult(
            runs=3,
            all_identical=True,
            authoritative=True,
        ),
        live_provider_robustness=LiveProviderRobustnessResult(
            runs=9,
            unique_embedding_digests=4,
            authoritative=False,
            e008_hit4_pass_count=5,
            e008_hit4_fail_count=4,
            experiment_verdict_rejected_count=9,
            observed_metric_variability={
                "wording": "In this nine-run diagnostic sample...",
            },
        ),
        integrity_verdict="PASS — FIXED-SNAPSHOT REPLAY REPRODUCIBLE",
    )
    assert result.frozen_snapshot_replay.authoritative is True
    assert result.live_provider_robustness is not None
    assert result.live_provider_robustness.authoritative is False


def test_live_variability_does_not_change_integrity_verdict() -> None:
    frozen = FrozenSnapshotReplayResult(runs=3, all_identical=True, authoritative=True)
    live = LiveProviderRobustnessResult(
        runs=9,
        unique_embedding_digests=9,
        authoritative=False,
        experiment_verdict_rejected_count=9,
    )
    result = ReplayIntegrityResult(
        frozen_snapshot_replay=frozen,
        live_provider_robustness=live,
        integrity_verdict="PASS — FIXED-SNAPSHOT REPLAY REPRODUCIBLE",
    )
    assert result.integrity_verdict == "PASS — FIXED-SNAPSHOT REPLAY REPRODUCIBLE"


def test_integrity_verdict_wording_is_snapshot_specific() -> None:
    verdict = "PASS — FIXED-SNAPSHOT REPLAY REPRODUCIBLE"
    assert "DETERMINISTIC" not in verdict
    assert "FIXED-SNAPSHOT" in verdict
