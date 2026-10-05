"""Derive the tracked retrieval acceptance summary from a pool-expansion evaluation artifact.

The raw per-case artifact of the final retrieval run (``data/05_evaluation/stage2g_pool_expansion.json``,
about 3 MB, gitignored like every generated evaluation output) is not committed. What is committed is
this derivation: the headline metrics, pool reachability, the comparison with the previously accepted
baseline, one compact row per frozen case and the deterministic customer-output behaviour for each
case, together with the identity of the run and the hash of the raw artifact it came from.

``build_summary`` is pure (dicts in, dict out) and unit-tested on synthetic artifacts;
``run_overlay`` and ``capture`` are the parts that need the application and the raw artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_JSON = "retrieval_summary.json"
SUMMARY_MD = "retrieval_summary.md"
SUMMARY_SCHEMA_VERSION = "1.0.0"
HISTORICAL_ARTIFACT_RELATIVE = "data/05_evaluation/vector_pool_expansion_v1.json"

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
RANKING_LIMITED = "candidate_generation_fixed_ranking_limited"
UNREACHABLE = "reachable_only_beyond_selected_pool"


# --- pure derivation -----------------------------------------------------------------------------------


def _ratio(rate: float, cases: int) -> dict[str, Any]:
    hits = round(rate * cases)
    if abs(hits / cases - rate) > 1e-9:
        raise ValueError(f"rate {rate} is not a whole number of {cases} cases")
    return {"value": round(rate, 4), "hits": hits, "cases": cases}


def _metrics(artifact: dict[str, Any]) -> dict[str, Any]:
    agg = artifact["candidate_ranking"]["aggregate_metrics"]
    sourced = agg["source_recall_case_count"]
    supporting_cases = agg["cases_with_supporting_documents"]
    return {
        "cases_total": agg["total_cases"],
        "cases_with_expected_sources": sourced,
        "hit_at_1": _ratio(agg["hit_rate_at_1"], sourced),
        "hit_at_4": _ratio(agg["hit_rate_at_4"], sourced),
        "hit_at_12": _ratio(agg["hit_rate_at_12"], sourced),
        "primary_hit_at_1": _ratio(agg["primary_source_hit_rate_at_1"], sourced),
        "primary_hit_at_4": _ratio(agg["primary_source_hit_rate_at_4"], sourced),
        "supporting_hit_at_4": {
            "value": round(agg["supporting_source_hit_rate_at_4"], 4),
            "hits": agg["supporting_source_hits_at_4"],
            "cases": supporting_cases,
        },
        "mrr": round(agg["mrr"], 4),
    }


def _reachability(artifact: dict[str, Any]) -> dict[str, Any]:
    reach = artifact["reachability_comparison"]
    slices = {item["risk_level"]: item for item in reach["risk_slices"]}
    return {
        "pool_at_12": {
            "primary": f"{reach['baseline_primary_reachable']}/{reach['baseline_primary_total']}",
            "supporting": f"{reach['baseline_supporting_reachable']}/{reach['baseline_supporting_total']}",
            "fully_unreachable": list(reach["baseline_fully_unreachable_cases"]),
        },
        "pool_at_24": {
            "primary": f"{reach['candidate_primary_reachable']}/{reach['candidate_primary_total']}",
            "supporting": f"{reach['candidate_supporting_reachable']}/{reach['candidate_supporting_total']}",
            "fully_unreachable": list(reach["candidate_fully_unreachable_cases"]),
        },
        "primary_by_risk": {
            level: {
                "pool_at_12": f"{slices[level]['baseline_primary_reachable']}/{slices[level]['baseline_primary_total']}",
                "pool_at_24": f"{slices[level]['candidate_primary_reachable']}/{slices[level]['candidate_primary_total']}",
            }
            for level in ("critical", "high", "medium", "low")
        },
    }


def _case_rows(
    artifact: dict[str, Any], overlay: dict[str, dict[str, Any]] | None
) -> list[dict[str, Any]]:
    rows = []
    for case in artifact["case_results"]:
        primary = set(case["expected_primary_documents"])
        final = case["candidate_ranking"]["ordered_candidates"]
        final_ranks = [i for i, item in enumerate(final, start=1) if item["document_id"] in primary]
        metrics = case["candidate_ranking"]["metrics"]
        row = {
            "case_id": case["case_id"],
            "risk": case["risk"],
            "category": case["category"],
            "expected_primary": [d.split("_")[0] for d in case["expected_primary_documents"]],
            "expected_supporting": [d.split("_")[0] for d in case["expected_supporting_documents"]],
            "primary_hit_at_4": metrics["primary_hit_at_4"],
            "hit_at_4": metrics["hit_at_4"],
            "reciprocal_rank": round(metrics["reciprocal_rank"], 4),
            "primary_best_final_rank": final_ranks[0] if final_ranks else None,
            "primary_best_pool24_rank": case["candidate_pool"]["reachability"]["primary_best_pool_rank"],
            "classification": case["failure_classification"],
        }
        if overlay is not None:
            row["deterministic"] = overlay[case["case_id"]]
        rows.append(row)
    return rows


def _compare_with_baseline(
    current: dict[str, Any], baseline: dict[str, Any], baseline_info: dict[str, Any]
) -> dict[str, Any]:
    cur_metrics, old_metrics = _metrics(current), _metrics(baseline)
    deltas = {}
    for key in ("hit_at_1", "hit_at_4", "hit_at_12", "primary_hit_at_1", "primary_hit_at_4"):
        deltas[key] = {
            "baseline": old_metrics[key]["value"],
            "current": cur_metrics[key]["value"],
            "delta": round(cur_metrics[key]["value"] - old_metrics[key]["value"], 4),
        }
    deltas["supporting_hit_at_4"] = {
        "baseline": old_metrics["supporting_hit_at_4"]["value"],
        "current": cur_metrics["supporting_hit_at_4"]["value"],
        "delta": round(
            cur_metrics["supporting_hit_at_4"]["value"] - old_metrics["supporting_hit_at_4"]["value"], 4
        ),
    }
    deltas["mrr"] = {
        "baseline": old_metrics["mrr"],
        "current": cur_metrics["mrr"],
        "delta": round(cur_metrics["mrr"] - old_metrics["mrr"], 4),
    }

    old_cases = {c["case_id"]: c for c in baseline["case_results"]}
    improvements: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    for case in current["case_results"]:
        before = old_cases[case["case_id"]]
        for key in ("primary_hit_at_4", "hit_at_4", "hit_at_1"):
            was = before["candidate_ranking"]["metrics"][key]
            now = case["candidate_ranking"]["metrics"][key]
            if was != now:
                entry = {"case_id": case["case_id"], "risk": case["risk"], "metric": key, "was": was, "now": now}
                (improvements if now else regressions).append(entry)
        was_reach = before["candidate_pool"]["reachability"]["primary_reachable"]
        now_reach = case["candidate_pool"]["reachability"]["primary_reachable"]
        if was_reach != now_reach:
            entry = {
                "case_id": case["case_id"],
                "risk": case["risk"],
                "metric": "pool24_primary_reachable",
                "was": was_reach,
                "now": now_reach,
            }
            (improvements if now_reach else regressions).append(entry)
    return {
        "baseline": baseline_info,
        "same_evaluation_dataset": (
            current["shared_context"]["evaluation_dataset_fingerprint"]
            == baseline["shared_context"]["evaluation_dataset_fingerprint"]
        ),
        "same_retrieval_config": current["experiment"]["config_hash"] == baseline["experiment"]["config_hash"],
        "same_reranker_config": (
            current["experiment"]["reranker_config_hash"] == baseline["experiment"]["reranker_config_hash"]
        ),
        "metric_deltas": deltas,
        "improvements": improvements,
        "regressions": regressions,
    }


def _limitations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unreachable = [r["case_id"] for r in rows if r["classification"] == UNREACHABLE]
    ranking_limited = [r["case_id"] for r in rows if r["classification"] == RANKING_LIMITED]
    missed = [r for r in rows if r["expected_primary"] and not r["primary_hit_at_4"]]
    limitations: dict[str, Any] = {
        "primary_outside_pool_at_24": unreachable,
        "ranking_limited_primary_in_pool_not_in_top4": ranking_limited,
        "primary_not_in_top4_by_risk": {
            level: [r["case_id"] for r in missed if r["risk"] == level] for level in ("critical", "high")
        },
    }
    if rows and "deterministic" in rows[0]:
        floors_below = [
            r["case_id"]
            for r in rows
            if r["risk"] in {"high", "critical"}
            and (
                r["deterministic"]["assessment_status"] != "rule_match"
                or RISK_ORDER[r["deterministic"]["risk_floor"]] < RISK_ORDER[r["risk"]]
            )
        ]
        over = [
            r["case_id"]
            for r in rows
            if r["deterministic"]["assessment_status"] == "rule_match"
            and RISK_ORDER[r["deterministic"]["risk_floor"]] - RISK_ORDER[r["risk"]] >= 2
        ]
        limitations["dataset_high_or_critical_without_matching_floor"] = floors_below
        limitations["deterministic_floor_two_or_more_levels_above_dataset_label"] = over
    return limitations


def build_summary(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    overlay: dict[str, dict[str, Any]] | None,
    derived_from: dict[str, Any],
    baseline_info: dict[str, Any],
) -> dict[str, Any]:
    shared = current["shared_context"]
    experiment = current["experiment"]
    rows = _case_rows(current, overlay)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "kind": "retrieval_acceptance_summary",
        "status": "current",
        "derived_from": derived_from,
        "identity": {
            "index_fingerprint": shared["index_fingerprint"],
            "chunk_count": shared["chunk_count"],
            "document_count": shared["document_count"],
            "collection": shared["collection"],
            "embedding_model": shared["embedding_model"],
            "evaluation_dataset_fingerprint": shared["evaluation_dataset_fingerprint"],
            "case_count": shared["case_count"],
            "retrieval_config_hash": experiment["config_hash"],
            "reranker_id": experiment["reranker_id"],
            "reranker_config_hash": experiment["reranker_config_hash"],
            "contract": {
                "fetch_k": shared["shared_fetch_k"],
                "pool_k": shared["candidate_pool_k"],
                "final_top_k": shared["final_top_k"],
                "threshold": shared["threshold"],
            },
        },
        "metrics": _metrics(current),
        "pool_reachability": _reachability(current),
        "comparison_with_previous_baseline": _compare_with_baseline(current, baseline, baseline_info),
        "limitations": _limitations(rows),
        "cases": rows,
    }


# --- rendering -----------------------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value:.3f}"


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def render_markdown(summary: dict[str, Any]) -> str:
    ident = summary["identity"]
    src = summary["derived_from"]
    metrics = summary["metrics"]
    reach = summary["pool_reachability"]
    cmp_ = summary["comparison_with_previous_baseline"]
    lim = summary["limitations"]
    rows = summary["cases"]
    lines = [
        "# Retrieval acceptance summary",
        "",
        "Derived from the final retrieval evaluation of the production index "
        f"(`{src['experiment_id']}`, mode `{src['experiment_mode']}`, real embeddings). "
        "Retrieval only: no model answers are scored here.",
        "",
        "## Provenance",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| Raw artifact | `{src['artifact']}` (not tracked: gitignored, ~{src['artifact_bytes'] // 1024} KiB) |",
        f"| Raw artifact SHA-256 | `{src['artifact_sha256']}` |",
        f"| Evaluation run (UTC) | {src['run_timestamp_utc']} |",
        f"| Evaluation code commit | `{src['run_git_commit']}` (working tree clean: {str(not src['run_git_dirty']).lower()}) |",
        f"| Summary derived by | `{src['derived_by']}` at `{src['derived_at_commit']}` |",
        "",
        "## Identity",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| Index / corpus fingerprint | `{ident['index_fingerprint']}` |",
        f"| Chunks / documents | {ident['chunk_count']} / {ident['document_count']} |",
        f"| Embedding model, collection | `{ident['embedding_model']}`, `{ident['collection']}` |",
        f"| Evaluation dataset fingerprint ({ident['case_count']} frozen cases) | `{ident['evaluation_dataset_fingerprint']}` |",
        f"| Retrieval config hash | `{ident['retrieval_config_hash']}` |",
        f"| Reranker | `{ident['reranker_id']}` (`{ident['reranker_config_hash']}`) |",
        f"| Contract | fetch {ident['contract']['fetch_k']} -> pool {ident['contract']['pool_k']} -> final "
        f"{ident['contract']['final_top_k']}, threshold {ident['contract']['threshold']} |",
        "",
        "## Metrics (production configuration: pool@24 -> source-authority-v1 -> final top-12)",
        "",
        "| Metric | Value | Cases |",
        "|--------|------:|------:|",
    ]
    for label, key in (
        ("Hit@1", "hit_at_1"),
        ("Hit@4", "hit_at_4"),
        ("Hit@12", "hit_at_12"),
        ("Primary hit@1", "primary_hit_at_1"),
        ("Primary hit@4", "primary_hit_at_4"),
        ("Supporting hit@4", "supporting_hit_at_4"),
    ):
        item = metrics[key]
        lines.append(f"| {label} | {_pct(item['value'])} | {item['hits']}/{item['cases']} |")
    lines += [
        f"| MRR | {_pct(metrics['mrr'])} | {metrics['cases_total']} |",
        "",
        "## Candidate-pool reachability of the expected primary document",
        "",
        "| Pool | Primary | Supporting | Fully unreachable |",
        "|------|--------:|-----------:|-------------------|",
        f"| pool@12 | {reach['pool_at_12']['primary']} | {reach['pool_at_12']['supporting']} | "
        f"{', '.join(reach['pool_at_12']['fully_unreachable']) or '-'} |",
        f"| pool@24 (production) | {reach['pool_at_24']['primary']} | {reach['pool_at_24']['supporting']} | "
        f"{', '.join(reach['pool_at_24']['fully_unreachable']) or '-'} |",
        "",
        "| Risk | pool@12 | pool@24 |",
        "|------|--------:|--------:|",
    ]
    for level in ("critical", "high", "medium", "low"):
        item = reach["primary_by_risk"][level]
        lines.append(f"| {level} | {item['pool_at_12']} | {item['pool_at_24']} |")

    base = cmp_["baseline"]
    lines += [
        "",
        "## Comparison with the previously accepted baseline",
        "",
        f"Baseline: `{base['artifact']}` (index `{base['index_fingerprint'][:8]}...`, {base['chunk_count']} chunks, "
        f"run {base['run_timestamp_utc']}), measured on the corpus **before** the content corrections that produced "
        "the current corpus. "
        f"Same frozen dataset: {str(cmp_['same_evaluation_dataset']).lower()}; same retrieval config: "
        f"{str(cmp_['same_retrieval_config']).lower()}; same reranker config: {str(cmp_['same_reranker_config']).lower()}.",
        "",
        "| Metric | Baseline | Current | Delta |",
        "|--------|---------:|--------:|------:|",
    ]
    for key, label in (
        ("hit_at_1", "Hit@1"),
        ("hit_at_4", "Hit@4"),
        ("hit_at_12", "Hit@12"),
        ("primary_hit_at_1", "Primary hit@1"),
        ("primary_hit_at_4", "Primary hit@4"),
        ("supporting_hit_at_4", "Supporting hit@4"),
        ("mrr", "MRR"),
    ):
        item = cmp_["metric_deltas"][key]
        lines.append(f"| {label} | {_pct(item['baseline'])} | {_pct(item['current'])} | {_signed(item['delta'])} |")
    lines += ["", f"Per-case regressions versus the baseline: **{len(cmp_['regressions'])}**."]
    if cmp_["regressions"]:
        for item in cmp_["regressions"]:
            lines.append(f"- {item['case_id']} ({item['risk']}): `{item['metric']}` {item['was']} -> {item['now']}")
    lines.append(f"Per-case improvements: **{len(cmp_['improvements'])}**.")
    for item in cmp_["improvements"]:
        lines.append(f"- {item['case_id']} ({item['risk']}): `{item['metric']}` {item['was']} -> {item['now']}")

    if rows and "deterministic" in rows[0]:
        lines += [
            "",
            "## High and critical cases: retrieval next to the deterministic customer-output behaviour",
            "",
            "`Primary@4` is whether the expected primary document is among the first four results. "
            "The deterministic columns are what the application does for the case text regardless of retrieval: "
            "the risk floor, and which text the customer-output policy selects when the model returns an "
            "ordinary, policy-clean grounded draft (`category_template` = fixed text that never comes from the "
            "knowledge base or the model). `no_signal` means no rule matched.",
            "",
            "| Case | Dataset risk | Primary@4 | Best primary rank (final top-12) | Assessment | Floor | Priority | Customer text |",
            "|------|--------------|:---------:|:--------------------------------:|------------|-------|:--------:|---------------|",
        ]
        for row in rows:
            if row["risk"] not in {"critical", "high"}:
                continue
            det = row["deterministic"]
            rank = row["primary_best_final_rank"] if row["primary_best_final_rank"] is not None else "-"
            lines.append(
                f"| {row['case_id']} | {row['risk']} | {'yes' if row['primary_hit_at_4'] else 'no'} | {rank} | "
                f"`{det['assessment_status']}` | `{det['risk_floor']}` | {'yes' if det['priority_handoff'] else 'no'} "
                f"| `{det['answer_provenance']}` |"
            )

    lines += [
        "",
        "## Known limitations visible in this run",
        "",
        f"- Expected primary document outside pool@24 for every candidate: {', '.join(lim['primary_outside_pool_at_24']) or 'none'}.",
        f"- Primary reachable in the pool but not in the first four (ranking-limited after pool expansion): "
        f"{', '.join(lim['ranking_limited_primary_in_pool_not_in_top4']) or 'none'}.",
        f"- Primary document not in the first four, critical cases: {', '.join(lim['primary_not_in_top4_by_risk']['critical']) or 'none'}; "
        f"high cases: {', '.join(lim['primary_not_in_top4_by_risk']['high']) or 'none'}.",
    ]
    if "dataset_high_or_critical_without_matching_floor" in lim:
        without = lim["dataset_high_or_critical_without_matching_floor"]
        over = lim["deterministic_floor_two_or_more_levels_above_dataset_label"]
        lines += [
            f"- Frozen cases labelled high or critical whose deterministic floor is lower or undetermined: "
            f"{', '.join(without) or 'none'} (reported as `no_signal`, never as an affirmative low risk).",
            f"- Deterministic floor two or more levels above the dataset label (over-escalation): "
            f"{', '.join(over) or 'none'}.",
        ]
    lines += [
        "",
        "Pool expansion improves candidate reachability, not top-4 ordering: the source-authority reranker's "
        "bonus is small by design, so a large similarity gap is not overcome. The experiment was frozen "
        "after it; see `docs/06_release_posture.md`.",
        "",
    ]
    return "\n".join(lines)


# --- application-dependent parts -------------------------------------------------------------------------


def sha256_raw(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_overlay(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Deterministic behaviour per case: the real pipeline over the artifact's final top-12 order.

    Retrieval is replayed from the artifact (chunk ids only, placeholder text), generation is a fixed
    ordinary grounded draft, so ``answer_provenance`` shows which text the customer-output policy
    selects for a policy-clean model draft; critical categories ignore the draft altogether.
    """
    from customer_claims_rag.application.customer_output import build_customer_output
    from customer_claims_rag.application.models import CustomerClaimsRequest
    from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
    from customer_claims_rag.generation.adapters.fake_chat import FakeChatModel
    from customer_claims_rag.generation.generator import GroundedGenerator
    from customer_claims_rag.generation.prompt_builder import PromptBuilder
    from customer_claims_rag.generation.risk_aware_generator import RiskAwareGroundedGenerator
    from customer_claims_rag.retrieval.models import SearchResult

    class ReplayedRetrieval:
        def __init__(self) -> None:
            self.final: list[dict[str, Any]] = []

        def search(self, query: str):
            return [
                SearchResult(
                    rank=rank,
                    chunk_id=item["chunk_id"],
                    document_id=item["document_id"],
                    content=item["chunk_id"],
                    source_path=f"data/02_clean_markdown/{item['document_id']}.md",
                    chunk_type=item["chunk_type"],
                    heading=item["chunk_id"],
                    heading_path=[item["chunk_id"]],
                    similarity=item["similarity"],
                    distance=item["distance"],
                )
                for rank, item in enumerate(self.final, start=1)
            ]

    retrieval = ReplayedRetrieval()
    draft = json.dumps(
        {"response_mode": "grounded_answer", "answer": "Сожалеем, что так вышло. Вопрос требует проверки. [S1]"},
        ensure_ascii=False,
    )
    generator = RiskAwareGroundedGenerator(
        grounded_generator=GroundedGenerator(
            chat_model=FakeChatModel(response=draft),
            prompt_builder=PromptBuilder(prompt_path=ROOT / "prompts" / "system_prompt.md"),
        )
    )
    pipeline = CustomerClaimsPipeline(retrieval=retrieval, generator=generator)

    overlay: dict[str, dict[str, Any]] = {}
    for case in artifact["case_results"]:
        retrieval.final = case["candidate_ranking"]["ordered_candidates"]
        output = build_customer_output(pipeline.handle(CustomerClaimsRequest(customer_query=case["query"])))
        overlay[case["case_id"]] = {
            "assessment_status": output.assessment_status.value,
            "risk_floor": output.risk_floor.value,
            "handoff_required": output.handoff_required,
            "priority_handoff": output.priority_handoff,
            "claim_category": output.claim_category,
            "answer_provenance": output.provenance,
        }
    return overlay


def capture(*, out_dir: Path, stage2g_artifact: Path, source_commit: str) -> tuple[list[Path], dict[str, Any]]:
    """Write ``retrieval_summary.json`` and ``.md``; return the files and the manifest's identity block."""
    if not stage2g_artifact.is_file():
        raise SystemExit(
            f"retrieval evaluation artifact not found: {stage2g_artifact}. It is a local, gitignored output of "
            "evaluate-pool-expansion; the committed summary is derived from it."
        )
    current = json.loads(stage2g_artifact.read_text(encoding="utf-8"))
    historical_path = ROOT / HISTORICAL_ARTIFACT_RELATIVE
    baseline = json.loads(historical_path.read_text(encoding="utf-8"))

    experiment = current["experiment"]
    derived_from = {
        "artifact": f"data/05_evaluation/{stage2g_artifact.name}",
        "artifact_sha256": sha256_raw(stage2g_artifact),
        "artifact_bytes": stage2g_artifact.stat().st_size,
        "tracked_in_git": False,
        "experiment_id": experiment["experiment_id"],
        "experiment_mode": experiment["experiment_mode"],
        "run_timestamp_utc": current["timestamp"],
        "run_git_commit": experiment["git_commit"],
        "run_git_dirty": experiment["git_dirty"],
        "derived_by": "scripts/summarize_retrieval_evidence.py",
        "derived_at_commit": source_commit,
    }
    baseline_info = {
        "artifact": HISTORICAL_ARTIFACT_RELATIVE,
        "artifact_sha256": sha256_raw(historical_path),
        "tracked_in_git": True,
        "index_fingerprint": baseline["shared_context"]["index_fingerprint"],
        "chunk_count": baseline["shared_context"]["chunk_count"],
        "run_timestamp_utc": baseline["timestamp"],
    }
    summary = build_summary(
        current,
        baseline,
        overlay=run_overlay(current),
        derived_from=derived_from,
        baseline_info=baseline_info,
    )
    json_path, md_path = out_dir / SUMMARY_JSON, out_dir / SUMMARY_MD
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    md_path.write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    metrics = summary["metrics"]
    identity = {
        "summary": SUMMARY_JSON,
        "index_fingerprint": summary["identity"]["index_fingerprint"],
        "evaluation_dataset_fingerprint": summary["identity"]["evaluation_dataset_fingerprint"],
        "retrieval_config_hash": summary["identity"]["retrieval_config_hash"],
        "raw_artifact_sha256": derived_from["artifact_sha256"],
        "hit_at_4": metrics["hit_at_4"]["value"],
        "primary_hit_at_4": metrics["primary_hit_at_4"]["value"],
        "mrr": metrics["mrr"],
    }
    return [json_path, md_path], identity
