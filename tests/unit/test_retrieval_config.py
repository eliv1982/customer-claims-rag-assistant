"""Unit tests for retrieval configuration."""

from __future__ import annotations

from pathlib import Path

from customer_claims_rag.application.settings import ApplicationSettings
from customer_claims_rag.generation_config import GenerationSettings
from customer_claims_rag.retrieval_config import RetrievalSettings
from tests.release_posture_helpers import resolved_release_target_for_index

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.md"
FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval" / "vector_pool_expansion_v1.json"
RERANKER_CONFIG_PATH = PROJECT_ROOT / "configs" / "reranking" / "source_authority_v1.json"


def _retrieval_settings(
    tmp_path: Path,
    *,
    api_key: str | None = "dummy-test-key",
) -> RetrievalSettings:
    return RetrievalSettings(
        index_dir=tmp_path / "index",
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
        top_k=4,
        fetch_k=12,
        similarity_threshold=0.35,
        embedding_batch_size=64,
        openai_api_key=api_key,
    )


def test_retrieval_settings_repr_does_not_leak_api_key(tmp_path: Path) -> None:
    secret = "sk-test-secret-value-12345"
    settings = _retrieval_settings(tmp_path, api_key=secret)
    rendered = repr(settings)
    assert secret not in rendered
    assert "sk-test" not in rendered


def test_retrieval_settings_repr_does_not_leak_partial_or_transformed_key(
    tmp_path: Path,
) -> None:
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"
    settings = _retrieval_settings(tmp_path, api_key=secret)
    rendered = repr(settings)
    assert secret not in rendered
    for fragment in ("sk-proj", "abcdef", "0123456789", secret[:8], secret[-8:]):
        assert fragment not in rendered


def test_retrieval_settings_repr_distinguishes_set_and_unset_key_states(
    tmp_path: Path,
) -> None:
    configured = repr(_retrieval_settings(tmp_path, api_key="sk-configured-key"))
    unconfigured = repr(_retrieval_settings(tmp_path, api_key=None))
    empty = repr(_retrieval_settings(tmp_path, api_key=""))

    assert "openai_api_key='set'" in configured
    assert "openai_api_key='unset'" in unconfigured
    assert "openai_api_key='unset'" in empty
    assert configured != unconfigured


def test_retrieval_settings_repr_includes_non_secret_fields(tmp_path: Path) -> None:
    settings = _retrieval_settings(tmp_path)
    rendered = repr(settings)

    assert f"index_dir={settings.index_dir!r}" in rendered
    assert "collection_name='customer_claims'" in rendered
    assert "embedding_model='text-embedding-3-small'" in rendered
    assert "top_k=4" in rendered
    assert "fetch_k=12" in rendered
    assert "similarity_threshold=0.35" in rendered
    assert "embedding_batch_size=64" in rendered


def test_application_settings_repr_remains_secret_safe(tmp_path: Path) -> None:
    secret = "sk-test-secret-value-12345"
    settings = ApplicationSettings(
        retrieval=_retrieval_settings(tmp_path, api_key=secret),
        generation=GenerationSettings(
            model_name="gpt-4o-mini",
            temperature=0.0,
            timeout_seconds=60.0,
            max_retries=2,
            max_output_tokens=1024,
            prompt_path=PROMPT_PATH.resolve(),
        ).validate(),
        frozen_retrieval_config_path=FROZEN_CONFIG_PATH,
        reranker_config_path=RERANKER_CONFIG_PATH,
        release_target=resolved_release_target_for_index(
            tmp_path / "index",
            project_root=tmp_path,
            corpus_fingerprint="fp",
            chunk_count=1,
            document_count=1,
            supported_document_ids=("01_service_overview",),
            embedding_model="text-embedding-3-small",
        ),
    )
    rendered = repr(settings)
    assert secret not in rendered
    assert "sk-test" not in rendered
    assert "openai_api_key='set'" in rendered
