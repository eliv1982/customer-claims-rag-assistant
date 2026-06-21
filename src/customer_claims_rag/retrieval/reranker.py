"""Deterministic source-authority rerankers for retrieval experiments."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from customer_claims_rag.retrieval.models import SearchResult


@dataclass(frozen=True)
class RerankerConfig:
    """Versioned reranker configuration loaded from JSON."""

    reranker_id: str
    version: str
    experiment_mode: str
    max_source_bonus: float
    source_authority_mapping: dict[str, float]
    tie_breaking: list[str]
    config_path: Path | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, config_path: Path | None = None) -> "RerankerConfig":
        return cls(
            reranker_id=str(payload["reranker_id"]),
            version=str(payload["version"]),
            experiment_mode=str(payload["experiment_mode"]),
            max_source_bonus=float(payload["max_source_bonus"]),
            source_authority_mapping={
                str(key): float(value)
                for key, value in payload["source_authority_mapping"].items()
            },
            tie_breaking=[str(item) for item in payload["tie_breaking"]],
            config_path=config_path,
        )


@dataclass(frozen=True)
class RerankedCandidate:
    """Single reranked candidate with audit components."""

    result: SearchResult
    baseline_rank: int
    candidate_rank: int
    source_authority_bonus: float
    rerank_score: float


def load_reranker_config(path: Path) -> RerankerConfig:
    """Load reranker config from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RerankerConfig.from_dict(payload, config_path=path.resolve())


def compute_config_hash(config: RerankerConfig) -> str:
    """Return SHA-256 hash of canonical config payload."""
    payload = {
        "reranker_id": config.reranker_id,
        "version": config.version,
        "experiment_mode": config.experiment_mode,
        "max_source_bonus": config.max_source_bonus,
        "source_authority_mapping": dict(
            sorted(config.source_authority_mapping.items())
        ),
        "tie_breaking": config.tie_breaking,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def source_authority_bonus(chunk_type: str | None, mapping: dict[str, float]) -> float:
    """Return bounded source authority bonus for a chunk type."""
    if not chunk_type:
        return 0.0
    return float(mapping.get(chunk_type, 0.0))


class SourceAuthorityV1Reranker:
    """Production-like reranker using chunk_type authority tiers only."""

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[RerankedCandidate]:
        """Rerank candidates without mutating inputs or similarity/distance."""
        _ = query  # reserved for future production-like signals; unused in v1
        if not candidates:
            return []

        working: list[tuple[SearchResult, int, float, float]] = []
        for candidate in candidates:
            copied = deepcopy(candidate)
            bonus = source_authority_bonus(copied.chunk_type, self.config.source_authority_mapping)
            rerank_score = copied.similarity + bonus
            working.append((copied, copied.rank, bonus, rerank_score))

        working.sort(
            key=lambda item: (
                -item[3],
                -item[0].similarity,
                item[1],
                item[0].chunk_id,
            )
        )

        reranked: list[RerankedCandidate] = []
        for candidate_rank, (result, baseline_rank, bonus, rerank_score) in enumerate(
            working,
            start=1,
        ):
            result.rank = candidate_rank
            reranked.append(
                RerankedCandidate(
                    result=result,
                    baseline_rank=baseline_rank,
                    candidate_rank=candidate_rank,
                    source_authority_bonus=bonus,
                    rerank_score=rerank_score,
                )
            )
        return reranked
