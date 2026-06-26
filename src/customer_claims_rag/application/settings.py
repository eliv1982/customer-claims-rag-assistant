"""Application-owned frozen retrieval configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

if TYPE_CHECKING:
    from customer_claims_rag.release.posture import ResolvedReleaseTarget

from customer_claims_rag import env_bootstrap
from customer_claims_rag.generation_config import GenerationSettings
from customer_claims_rag.retrieval_config import (
    RetrievalSettings,
    validate_fetch_k,
    validate_similarity_threshold,
    validate_top_k,
)

DEFAULT_FROZEN_RETRIEVAL_CONFIG_RELATIVE = Path("configs") / "retrieval" / "vector_pool_expansion_v1.json"
DEFAULT_RERANKER_CONFIG_RELATIVE = Path("configs") / "reranking" / "source_authority_v1.json"
DEFAULT_FROZEN_RETRIEVAL_CONFIG_PATH = (
    env_bootstrap.project_root() / DEFAULT_FROZEN_RETRIEVAL_CONFIG_RELATIVE
)
DEFAULT_RERANKER_CONFIG_PATH = (
    env_bootstrap.project_root() / DEFAULT_RERANKER_CONFIG_RELATIVE
)


@dataclass(frozen=True)
class ApplicationSettings:
    """Aggregate application settings for production pipeline wiring."""

    retrieval: RetrievalSettings
    generation: GenerationSettings
    frozen_retrieval_config_path: Path
    reranker_config_path: Path
    release_target: ResolvedReleaseTarget
    release_descriptor_path: Path | None = None

    def __repr__(self) -> str:
        api_key_state = "set" if self.retrieval.openai_api_key else "unset"
        return (
            "ApplicationSettings("
            f"retrieval_index_dir={self.retrieval.index_dir!r}, "
            f"retrieval_collection_name={self.retrieval.collection_name!r}, "
            f"retrieval_embedding_model={self.retrieval.embedding_model!r}, "
            f"release_target={self.release_target.target_name!r}, "
            f"openai_api_key={api_key_state!r}, "
            f"generation_model_name={self.generation.model_name!r}, "
            f"frozen_retrieval_config_path={self.frozen_retrieval_config_path!r}, "
            f"reranker_config_path={self.reranker_config_path!r})"
        )

    @classmethod
    def from_env(cls) -> ApplicationSettings:
        from customer_claims_rag.release.posture import load_production_release_context

        env_bootstrap.load_project_env()
        root = env_bootstrap.project_root()
        _descriptor, resolved_target = load_production_release_context(project_root=root)
        retrieval = RetrievalSettings.from_env()
        retrieval = replace(
            retrieval,
            index_dir=resolved_target.index_dir,
            collection_name=resolved_target.collection_name,
            embedding_model=resolved_target.embedding_model,
        )
        return cls(
            retrieval=retrieval,
            generation=GenerationSettings.from_env(),
            frozen_retrieval_config_path=root / DEFAULT_FROZEN_RETRIEVAL_CONFIG_RELATIVE,
            reranker_config_path=root / DEFAULT_RERANKER_CONFIG_RELATIVE,
            release_target=resolved_target,
            release_descriptor_path=None,
        )


_KNOWN_FROZEN_RETRIEVAL_CONFIG_KEYS = frozenset(
    {
        "experiment_id",
        "version",
        "experiment_mode",
        "baseline_pool_k",
        "candidate_pool_k",
        "final_top_k",
        "threshold",
        "reranker_id",
        "reranker_config_hash",
        "tie_breaking",
    },
)


class _FrozenRetrievalConfigFile(BaseModel):
    """Exact on-disk schema for the frozen retrieval JSON artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    experiment_id: str
    version: str
    experiment_mode: str
    baseline_pool_k: int
    candidate_pool_k: int
    final_top_k: int
    threshold: float
    reranker_id: str
    reranker_config_hash: str
    tie_breaking: list[str]

    @field_validator("experiment_id", "version", "reranker_id")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be a non-empty string")
        return value


class FrozenRetrievalConfig(BaseModel):
    """Normalized frozen retrieval policy for production application wiring."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    config_id: str
    version: str
    vector_top_k: int
    vector_fetch_k: int
    similarity_threshold: float
    candidate_pool_k: int
    final_top_k: int
    reranker_id: str

    @field_validator("config_id", "version", "reranker_id")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be a non-empty string")
        return value

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        validate_top_k(self.vector_top_k)
        validate_fetch_k(self.vector_fetch_k, self.vector_top_k)
        validate_similarity_threshold(self.similarity_threshold)

        if self.candidate_pool_k < 1:
            raise ValueError("candidate_pool_k must be >= 1")
        if self.final_top_k < 1:
            raise ValueError("final_top_k must be >= 1")
        if self.final_top_k > self.candidate_pool_k:
            raise ValueError("final_top_k must be <= candidate_pool_k")
        if self.candidate_pool_k > self.vector_top_k:
            raise ValueError("candidate_pool_k must be <= vector_top_k")
        if self.vector_top_k != self.candidate_pool_k:
            raise ValueError("vector_top_k must equal candidate_pool_k")
        if self.vector_fetch_k != self.candidate_pool_k:
            raise ValueError("vector_fetch_k must equal candidate_pool_k")

        return self


def load_frozen_retrieval_config(path: Path) -> FrozenRetrievalConfig:
    """Load and normalize the frozen retrieval JSON config from disk."""
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError("frozen retrieval config must be a JSON object")

    unknown_keys = set(raw_payload) - _KNOWN_FROZEN_RETRIEVAL_CONFIG_KEYS
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"unknown top-level frozen retrieval config keys: {unknown}")

    parsed = _FrozenRetrievalConfigFile.model_validate(raw_payload)
    pool_k = parsed.candidate_pool_k
    return FrozenRetrievalConfig(
        config_id=parsed.experiment_id,
        version=parsed.version,
        vector_top_k=pool_k,
        vector_fetch_k=pool_k,
        similarity_threshold=parsed.threshold,
        candidate_pool_k=pool_k,
        final_top_k=parsed.final_top_k,
        reranker_id=parsed.reranker_id,
    )
