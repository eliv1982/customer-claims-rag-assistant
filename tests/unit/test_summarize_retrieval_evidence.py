"""The retrieval acceptance summary is derived, not hand-written: pin the derivation.

``scripts/summarize_retrieval_evidence.py`` turns the (untracked) evaluation artifact into the committed
``deliverables/evidence/retrieval_summary.*``. The pure part is exercised here on small synthetic
artifacts, including the case that matters for an acceptance claim: a per-case regression against the
previous baseline must be *seen*, never averaged away.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "summarize_retrieval_evidence.py"


def _load():
    spec = importlib.util.spec_from_file_location("summarize_retrieval_evidence_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


summary_tool = _load()

# (case_id, risk, expected primary, primary in top 4, primary in pool@24, classification)
CASES = (
    ("T001", "low", ["01_service_overview"], True, True, "none"),
    ("T002", "critical", ["08_escalation_and_risk_rules"], True, True, "already_reachable_at_baseline"),
    ("T003", "critical", ["06_food_quality_and_packaging"], False, True, summary_tool.RANKING_LIMITED),
    ("T004", "high", ["07_complaint_handling_procedure"], False, False, summary_tool.UNREACHABLE),
    ("T005", "high", ["02_delivery_rules"], True, True, "none"),
)


def _case(case_id, risk, primary, in_top4, in_pool, classification) -> dict:
    ordered = [{"document_id": "99_other", "chunk_id": "99_other::chunk-001"}] * 3
    ordered.insert(0 if in_top4 else 5, {"document_id": primary[0], "chunk_id": f"{primary[0]}::chunk-001"})
    return {
        "case_id": case_id,
        "risk": risk,
        "category": "quality",
        "query": "text",
        "expected_primary_documents": primary,
        "expected_supporting_documents": [],
        "failure_classification": classification,
        "candidate_pool": {
            "reachability": {"primary_reachable": in_pool, "primary_best_pool_rank": 3 if in_pool else None}
        },
        "candidate_ranking": {
            "ordered_candidates": ordered,
            "metrics": {
                "primary_hit_at_4": in_top4,
                "hit_at_4": in_top4,
                "hit_at_1": in_top4,
                "reciprocal_rank": 1.0 if in_top4 else 0.0,
            },
        },
    }


def _artifact(*, fingerprint="a" * 64, cases=CASES, dataset="d" * 64) -> dict:
    rows = [_case(*case) for case in cases]
    n = len(rows)
    top4 = sum(1 for r in rows if r["candidate_ranking"]["metrics"]["primary_hit_at_4"])
    slices = []
    for level in ("critical", "high", "medium", "low"):
        scoped = [r for r in rows if r["risk"] == level]
        reachable = sum(1 for r in scoped if r["candidate_pool"]["reachability"]["primary_reachable"])
        slices.append(
            {
                "risk_level": level,
                "case_count": len(scoped),
                "baseline_primary_reachable": reachable,
                "candidate_primary_reachable": reachable,
                "baseline_primary_total": len(scoped),
                "candidate_primary_total": len(scoped),
            }
        )
    reachable_all = sum(1 for r in rows if r["candidate_pool"]["reachability"]["primary_reachable"])
    rate = top4 / n
    return {
        "timestamp": "2026-01-01T00:00:00Z",
        "experiment": {
            "experiment_id": "vector-pool-expansion-v1",
            "experiment_mode": "production-like",
            "config_hash": "c" * 64,
            "reranker_id": "source-authority-v1",
            "reranker_config_hash": "r" * 64,
            "git_commit": "b" * 40,
            "git_dirty": False,
        },
        "shared_context": {
            "index_fingerprint": fingerprint,
            "chunk_count": 216,
            "document_count": 10,
            "collection": "customer_claims",
            "embedding_model": "text-embedding-3-small",
            "evaluation_dataset_fingerprint": dataset,
            "case_count": n,
            "shared_fetch_k": 24,
            "candidate_pool_k": 24,
            "final_top_k": 12,
            "threshold": 0.0,
        },
        "candidate_ranking": {
            "aggregate_metrics": {
                "total_cases": n,
                "source_recall_case_count": n,
                "cases_with_supporting_documents": 0,
                "hit_rate_at_1": rate,
                "hit_rate_at_4": rate,
                "hit_rate_at_12": rate,
                "primary_source_hit_rate_at_1": rate,
                "primary_source_hit_rate_at_4": rate,
                "supporting_source_hit_rate_at_4": 0.0,
                "supporting_source_hits_at_4": 0,
                "mrr": rate,
            }
        },
        "reachability_comparison": {
            "baseline_primary_reachable": reachable_all,
            "baseline_primary_total": n,
            "baseline_supporting_reachable": 0,
            "baseline_supporting_total": 0,
            "baseline_fully_unreachable_cases": [r["case_id"] for r in rows if not r["candidate_pool"]["reachability"]["primary_reachable"]],
            "candidate_primary_reachable": reachable_all,
            "candidate_primary_total": n,
            "candidate_supporting_reachable": 0,
            "candidate_supporting_total": 0,
            "candidate_fully_unreachable_cases": [r["case_id"] for r in rows if not r["candidate_pool"]["reachability"]["primary_reachable"]],
            "risk_slices": slices,
        },
        "case_results": rows,
    }


DERIVED = {
    "artifact": "data/05_evaluation/x.json",
    "artifact_sha256": "0" * 64,
    "artifact_bytes": 2048,
    "tracked_in_git": False,
    "experiment_id": "vector-pool-expansion-v1",
    "experiment_mode": "production-like",
    "run_timestamp_utc": "2026-01-01T00:00:00Z",
    "run_git_commit": "b" * 40,
    "run_git_dirty": False,
    "derived_by": "scripts/summarize_retrieval_evidence.py",
    "derived_at_commit": "e" * 40,
}
BASELINE_INFO = {
    "artifact": "data/05_evaluation/old.json",
    "artifact_sha256": "1" * 64,
    "tracked_in_git": True,
    "index_fingerprint": "f" * 64,
    "chunk_count": 215,
    "run_timestamp_utc": "2025-01-01T00:00:00Z",
}


def _summary(current: dict, baseline: dict, overlay=None) -> dict:
    return summary_tool.build_summary(
        current, baseline, overlay=overlay, derived_from=DERIVED, baseline_info=BASELINE_INFO
    )


def test_metrics_are_whole_case_counts_with_their_denominators() -> None:
    summary = _summary(_artifact(), _artifact(fingerprint="f" * 64))
    assert summary["metrics"]["primary_hit_at_4"] == {"value": 0.6, "hits": 3, "cases": 5}
    assert summary["metrics"]["hit_at_4"]["hits"] == 3
    with pytest.raises(ValueError, match="whole number"):
        summary_tool._ratio(0.61, 5)


def test_reachability_is_reported_per_pool_and_per_risk() -> None:
    reach = _summary(_artifact(), _artifact())["pool_reachability"]
    assert reach["pool_at_24"]["primary"] == "4/5"
    assert reach["pool_at_24"]["fully_unreachable"] == ["T004"]
    assert reach["primary_by_risk"]["critical"]["pool_at_24"] == "2/2"
    assert reach["primary_by_risk"]["high"]["pool_at_24"] == "1/2"


def test_an_unchanged_run_has_no_improvements_and_no_regressions() -> None:
    comparison = _summary(_artifact(), _artifact(fingerprint="f" * 64))["comparison_with_previous_baseline"]
    assert comparison["improvements"] == [] and comparison["regressions"] == []
    assert comparison["same_evaluation_dataset"] and comparison["same_retrieval_config"]
    assert comparison["metric_deltas"]["primary_hit_at_4"]["delta"] == 0.0


def test_a_per_case_improvement_is_reported() -> None:
    worse = tuple(
        ("T005", "high", ["02_delivery_rules"], False, True, "none") if case[0] == "T005" else case
        for case in CASES
    )
    comparison = _summary(_artifact(), _artifact(cases=worse))["comparison_with_previous_baseline"]
    assert [item["case_id"] for item in comparison["improvements"]] == ["T005"] * 3  # primary@4, hit@4, hit@1
    assert comparison["regressions"] == []
    assert comparison["metric_deltas"]["primary_hit_at_4"]["delta"] == pytest.approx(0.2)


def test_a_per_case_regression_is_never_averaged_away() -> None:
    """A critical case that was a hit and is not any more must be listed even if another case improved."""
    previous = tuple(
        ("T001", "low", ["01_service_overview"], False, True, "none")  # T001 improves ...
        if case[0] == "T001"
        else case
        for case in CASES
    )
    current = tuple(
        ("T002", "critical", ["08_escalation_and_risk_rules"], False, True, "none")  # ... T002 regresses
        if case[0] == "T002"
        else case
        for case in CASES
    )
    comparison = _summary(_artifact(cases=current), _artifact(cases=previous))["comparison_with_previous_baseline"]
    assert comparison["metric_deltas"]["primary_hit_at_4"]["delta"] == 0.0  # the aggregate hides it
    assert [item["case_id"] for item in comparison["regressions"]] == ["T002"] * 3  # primary, hit@4, hit@1
    assert comparison["regressions"][0]["risk"] == "critical"
    assert [item["case_id"] for item in comparison["improvements"]] == ["T001"] * 3


def test_pool_reachability_loss_is_a_regression() -> None:
    previous = _artifact()
    current_cases = tuple(
        ("T005", "high", ["02_delivery_rules"], False, False, summary_tool.UNREACHABLE) if case[0] == "T005" else case
        for case in CASES
    )
    comparison = _summary(_artifact(cases=current_cases), previous)["comparison_with_previous_baseline"]
    assert {"metric": "pool24_primary_reachable", "was": True, "now": False}.items() <= next(
        item for item in comparison["regressions"] if item["metric"] == "pool24_primary_reachable"
    ).items()


def test_a_different_dataset_or_config_is_flagged_not_hidden() -> None:
    other = _artifact(dataset="e" * 64)
    other["experiment"]["config_hash"] = "9" * 64
    comparison = _summary(_artifact(), other)["comparison_with_previous_baseline"]
    assert comparison["same_evaluation_dataset"] is False
    assert comparison["same_retrieval_config"] is False


def test_limitations_are_derived_from_the_cases() -> None:
    limitations = _summary(_artifact(), _artifact())["limitations"]
    assert limitations["primary_outside_pool_at_24"] == ["T004"]
    assert limitations["ranking_limited_primary_in_pool_not_in_top4"] == ["T003"]
    assert limitations["primary_not_in_top4_by_risk"] == {"critical": ["T003"], "high": ["T004"]}


def _overlay(**overrides) -> dict:
    base = {
        "assessment_status": "rule_match",
        "risk_floor": "critical",
        "handoff_required": True,
        "priority_handoff": True,
        "claim_category": "x",
        "answer_provenance": "category_template",
    }
    rows = {case[0]: dict(base) for case in CASES}
    rows["T001"] |= {"assessment_status": "no_signal", "risk_floor": "low", "priority_handoff": False}
    for case_id, change in overrides.items():
        rows[case_id] |= change
    return rows


def test_rule_gaps_and_over_escalation_are_reported_with_the_overlay() -> None:
    overlay = _overlay(
        T005={"risk_floor": "low", "assessment_status": "no_signal"},  # dataset high, rules silent
        T001={"risk_floor": "critical", "assessment_status": "rule_match"},  # dataset low, floor critical
    )
    limitations = _summary(_artifact(), _artifact(), overlay=overlay)["limitations"]
    assert limitations["dataset_high_or_critical_without_matching_floor"] == ["T005"]
    assert limitations["deterministic_floor_two_or_more_levels_above_dataset_label"] == ["T001"]


def test_the_markdown_states_identity_provenance_and_the_comparison() -> None:
    summary = _summary(_artifact(), _artifact(fingerprint="f" * 64), overlay=_overlay())
    text = summary_tool.render_markdown(summary)
    for needle in (
        "a" * 64,  # index fingerprint
        "0" * 64,  # raw artifact hash
        "Per-case regressions versus the baseline: **0**",
        "| Primary hit@4 | 0.600 | 3/5 |",
        "## High and critical cases",
        "T003",
    ):
        assert needle in text, needle
    assert "absolute" not in text.lower()


def test_summary_json_round_trips_and_carries_no_workstation_paths() -> None:
    text = json.dumps(_summary(_artifact(), _artifact(), overlay=_overlay()), ensure_ascii=False)
    assert "\\Users\\" not in text and "/Users/" not in text and "/home/" not in text
    assert json.loads(text)["schema_version"] == summary_tool.SUMMARY_SCHEMA_VERSION


def test_deriving_does_not_mutate_its_inputs() -> None:
    current, baseline = _artifact(), _artifact()
    before = copy.deepcopy((current, baseline))
    _summary(current, baseline, overlay=_overlay())
    assert (current, baseline) == before
