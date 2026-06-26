"""Unit tests for application settings and production factory wiring."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from customer_claims_rag.application.factory import build_customer_claims_pipeline
from customer_claims_rag.application.frozen_retrieval import FrozenRetrievalService
from customer_claims_rag.application.models import CustomerClaimsRequest
from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
from customer_claims_rag.application.settings import (
    ApplicationSettings,
    FrozenRetrievalConfig,
    load_frozen_retrieval_config,
)
from customer_claims_rag.exceptions import (
    EmbeddingError,
    GenerationError,
    IndexManifestError,
    ReleasePostureError,
    RetrievalError,
)
from customer_claims_rag.generation.generator import GroundedGenerator
from customer_claims_rag.generation.risk_aware_generator import RiskAwareGroundedGenerator
from customer_claims_rag.generation_config import GenerationSettings
from customer_claims_rag.retrieval.retriever import BaselineRetriever
from customer_claims_rag.retrieval_config import RetrievalSettings
from customer_claims_rag.release.posture import ReleasePostureDiagnostics
from tests.release_posture_helpers import (
    resolved_release_target_for_index,
    write_test_release_descriptor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "grounded_answer_v1.md"
FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval" / "vector_pool_expansion_v1.json"
RERANKER_CONFIG_PATH = PROJECT_ROOT / "configs" / "reranking" / "source_authority_v1.json"


def _retrieval_settings(
    tmp_path: Path,
    *,
    top_k: int = 4,
    fetch_k: int = 12,
    api_key: str | None = "dummy-test-key",
) -> RetrievalSettings:
    return RetrievalSettings(
        index_dir=tmp_path / "index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        top_k=top_k,
        fetch_k=fetch_k,
        similarity_threshold=0.0,
        embedding_batch_size=64,
        openai_api_key=api_key,
    )


def _generation_settings() -> GenerationSettings:
    return GenerationSettings(
        model_name="gpt-4o-mini",
        temperature=0.0,
        timeout_seconds=60.0,
        max_retries=2,
        max_output_tokens=1024,
        prompt_path=PROMPT_PATH.resolve(),
    ).validate()


def _application_settings(
    tmp_path: Path,
    *,
    frozen_config_path: Path = FROZEN_CONFIG_PATH,
    reranker_config_path: Path = RERANKER_CONFIG_PATH,
    retrieval: RetrievalSettings | None = None,
    index_dir: Path | None = None,
) -> ApplicationSettings:
    resolved_retrieval = retrieval or _retrieval_settings(tmp_path)
    if index_dir is not None:
        resolved_retrieval = RetrievalSettings(
            index_dir=index_dir,
            collection_name=resolved_retrieval.collection_name,
            embedding_model=resolved_retrieval.embedding_model,
            top_k=resolved_retrieval.top_k,
            fetch_k=resolved_retrieval.fetch_k,
            similarity_threshold=resolved_retrieval.similarity_threshold,
            embedding_batch_size=resolved_retrieval.embedding_batch_size,
            openai_api_key=resolved_retrieval.openai_api_key,
        )
    descriptor_path = tmp_path / "release_descriptor.json"
    write_test_release_descriptor(
        descriptor_path,
        index_path_relative="index",
        corpus_fingerprint="test-fingerprint",
        chunk_count=1,
        document_count=1,
        supported_document_ids=["01_service_overview"],
    )
    release_target = resolved_release_target_for_index(
        resolved_retrieval.index_dir,
        project_root=tmp_path,
        corpus_fingerprint="test-fingerprint",
        chunk_count=1,
        document_count=1,
        supported_document_ids=("01_service_overview",),
        embedding_model=resolved_retrieval.embedding_model,
    )
    return ApplicationSettings(
        retrieval=resolved_retrieval,
        generation=_generation_settings(),
        frozen_retrieval_config_path=frozen_config_path,
        reranker_config_path=reranker_config_path,
        release_target=release_target,
        release_descriptor_path=descriptor_path,
    )


def _frozen_config() -> FrozenRetrievalConfig:
    return load_frozen_retrieval_config(FROZEN_CONFIG_PATH)


def test_application_settings_explicit_construction(tmp_path: Path) -> None:
    settings = _application_settings(tmp_path)
    assert settings.frozen_retrieval_config_path == FROZEN_CONFIG_PATH
    assert settings.reranker_config_path == RERANKER_CONFIG_PATH
    assert settings.retrieval.embedding_model == "fake-embedding-model"
    assert settings.generation.model_name == "gpt-4o-mini"


def test_application_settings_is_frozen(tmp_path: Path) -> None:
    settings = _application_settings(tmp_path)
    with pytest.raises(AttributeError):
        settings.frozen_retrieval_config_path = RERANKER_CONFIG_PATH  # type: ignore[misc]


def test_application_settings_paths_are_path_objects(tmp_path: Path) -> None:
    settings = _application_settings(tmp_path)
    assert isinstance(settings.frozen_retrieval_config_path, Path)
    assert isinstance(settings.reranker_config_path, Path)
    assert isinstance(settings.retrieval.index_dir, Path)
    assert isinstance(settings.generation.prompt_path, Path)


def test_application_settings_does_not_duplicate_nested_fields(tmp_path: Path) -> None:
    settings = _application_settings(tmp_path)
    assert not hasattr(settings, "openai_api_key")
    assert not hasattr(settings, "embedding_model")
    assert not hasattr(settings, "model_name")
    assert not hasattr(settings, "index_dir")


def test_application_settings_repr_does_not_leak_api_key(tmp_path: Path) -> None:
    secret = "sk-test-secret-value-12345"
    settings = _application_settings(
        tmp_path,
        retrieval=_retrieval_settings(tmp_path, api_key=secret),
    )
    rendered = repr(settings)
    assert secret not in rendered
    assert "sk-test" not in rendered


def test_settings_module_import_does_not_call_load_project_env() -> None:
    source = (
        PROJECT_ROOT / "src" / "customer_claims_rag" / "application" / "settings.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    from_env = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ApplicationSettings":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "from_env":
                    from_env = item
    assert from_env is not None
    from_env_source = ast.get_source_segment(source, from_env) or ""
    module_without_from_env = source.replace(from_env_source, "")
    assert "load_project_env" not in module_without_from_env


def test_factory_module_import_does_not_call_load_project_env() -> None:
    source = (
        PROJECT_ROOT / "src" / "customer_claims_rag" / "application" / "factory.py"
    ).read_text(encoding="utf-8")
    assert "load_project_env" not in source


def test_application_settings_from_env_reuses_nested_loaders(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from customer_claims_rag import env_bootstrap
    from customer_claims_rag.release.posture import ResolvedReleaseTarget

    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    env_bootstrap.reset_project_env()
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    release_target = ResolvedReleaseTarget(
        target_name="active",
        status="selected_production_release",
        index_dir=tmp_path / "release-index",
        index_path_relative="release-index",
        expected_corpus_fingerprint="fp",
        expected_chunk_count=1,
        expected_document_count=1,
        supported_document_ids=("01_service_overview",),
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
        vector_dimension=1536,
    )
    retrieval_settings = _retrieval_settings(tmp_path)
    retrieval_mock = MagicMock(return_value=retrieval_settings)
    generation_mock = MagicMock(return_value=_generation_settings())
    monkeypatch.setattr(
        "customer_claims_rag.application.settings.RetrievalSettings.from_env",
        retrieval_mock,
    )
    monkeypatch.setattr(
        "customer_claims_rag.application.settings.GenerationSettings.from_env",
        generation_mock,
    )
    monkeypatch.setattr(
        "customer_claims_rag.release.posture.load_production_release_context",
        lambda **_: (MagicMock(), release_target),
    )

    settings = ApplicationSettings.from_env()

    retrieval_mock.assert_called_once_with()
    generation_mock.assert_called_once_with()
    assert settings.retrieval.index_dir == release_target.index_dir
    assert settings.release_target is release_target
    assert settings.frozen_retrieval_config_path == (
        tmp_path / "configs" / "retrieval" / "vector_pool_expansion_v1.json"
    )
    assert settings.reranker_config_path == (
        tmp_path / "configs" / "reranking" / "source_authority_v1.json"
    )


def test_default_config_paths_use_project_root_not_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from customer_claims_rag import env_bootstrap

    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    env_bootstrap.reset_project_env()
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir(parents=True)
    monkeypatch.chdir(other_cwd)

    monkeypatch.setattr(
        "customer_claims_rag.application.settings.RetrievalSettings.from_env",
        lambda: _retrieval_settings(tmp_path),
    )
    monkeypatch.setattr(
        "customer_claims_rag.application.settings.GenerationSettings.from_env",
        _generation_settings,
    )
    monkeypatch.setattr(
        "customer_claims_rag.release.posture.load_production_release_context",
        lambda **_: (
            MagicMock(),
            resolved_release_target_for_index(
                tmp_path / "release-index",
                project_root=tmp_path,
                corpus_fingerprint="fp",
                chunk_count=1,
                document_count=1,
                supported_document_ids=("01_service_overview",),
                embedding_model="fake-embedding-model",
            ),
        ),
    )

    settings = ApplicationSettings.from_env()
    assert settings.frozen_retrieval_config_path == (
        tmp_path / "configs" / "retrieval" / "vector_pool_expansion_v1.json"
    )
    assert settings.reranker_config_path == (
        tmp_path / "configs" / "reranking" / "source_authority_v1.json"
    )


def test_application_settings_from_env_missing_api_key_follows_retrieval_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from customer_claims_rag import env_bootstrap

    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    env_bootstrap.reset_project_env()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_bootstrap.load_project_env(force=True)

    monkeypatch.setattr(
        "customer_claims_rag.release.posture.load_production_release_context",
        lambda **_: (
            MagicMock(),
            resolved_release_target_for_index(
                tmp_path / "release-index",
                project_root=tmp_path,
                corpus_fingerprint="fp",
                chunk_count=1,
                document_count=1,
                supported_document_ids=("01_service_overview",),
                embedding_model="fake-embedding-model",
            ),
        ),
    )

    settings = ApplicationSettings.from_env()
    assert settings.retrieval.openai_api_key is None


class _RecordingEmbeddingProvider:
    model_name = "fake-embedding-model"

    def embed_documents(self, texts):
        return []

    def embed_query(self, text):
        return [0.0] * 8


class _RecordingVectorStore:
    collection_name = "customer_claims"

    def count(self) -> int:
        return 0

    def recreate_collection(self, *, embedding_dimension: int) -> None:
        pass

    def add_chunks(self, chunks, embeddings) -> None:
        pass

    def similarity_search(self, query_embedding, *, k: int):
        return []

    def close(self) -> None:
        pass


def _build_pipeline_with_recorders(
    tmp_path: Path,
    *,
    validate_index: MagicMock | None = None,
    frozen_config_loader=load_frozen_retrieval_config,
    reranker_config_loader=None,
    embedding_factory=None,
    vector_store_factory=None,
    grounded_generator_factory=None,
    settings: ApplicationSettings | None = None,
):
    if reranker_config_loader is None:
        from customer_claims_rag.retrieval.reranker import load_reranker_config

        reranker_config_loader = load_reranker_config

    embedding_calls: list[dict] = []
    vector_store_calls: list[dict] = []
    generator_calls: list[GenerationSettings] = []

    def _embedding_factory(**kwargs):
        embedding_calls.append(kwargs)
        if embedding_factory is not None:
            return embedding_factory(**kwargs)
        return _RecordingEmbeddingProvider()

    def _vector_store_factory(**kwargs):
        vector_store_calls.append(kwargs)
        if vector_store_factory is not None:
            return vector_store_factory(**kwargs)
        return _RecordingVectorStore()

    def _grounded_generator_factory(generation_settings: GenerationSettings) -> GroundedGenerator:
        generator_calls.append(generation_settings)
        if grounded_generator_factory is not None:
            return grounded_generator_factory(generation_settings)
        return MagicMock(spec=GroundedGenerator)

    validate_mock = validate_index or MagicMock()

    diagnostics = ReleasePostureDiagnostics(
        release_posture_id="test",
        selected_target="active",
        target_status="selected_production_release",
        index_path_relative="index",
        corpus_fingerprint="test-fingerprint",
        chunk_count=1,
        document_count=1,
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        vector_dimension=8,
        frozen_retrieval_config_path="configs/retrieval/vector_pool_expansion_v1.json",
        frozen_config_hash="ff53ff9721ad86b1c542bf96dce616d9057ed3b347e341fed59750b07b69e048",
        vector_fetch_k=24,
        candidate_pool_k=24,
        final_top_k=12,
        similarity_threshold=0.0,
        reranker_id="source-authority-v1",
    )

    with patch(
        "customer_claims_rag.application.factory.validate_release_posture_for_production",
        return_value=diagnostics,
    ), patch(
        "customer_claims_rag.application.factory.load_release_posture_descriptor",
        return_value=MagicMock(),
    ), patch(
        "customer_claims_rag.application.factory.BaselineRetriever"
    ) as retriever_cls:
        retriever_instance = MagicMock()
        retriever_instance.validate_index = validate_mock
        retriever_cls.return_value = retriever_instance

        pipeline = build_customer_claims_pipeline(
            settings or _application_settings(tmp_path),
            embedding_provider_factory=_embedding_factory,
            vector_store_factory=_vector_store_factory,
            grounded_generator_factory=_grounded_generator_factory,
            frozen_config_loader=frozen_config_loader,
            reranker_config_loader=reranker_config_loader,
        )

    return (
        pipeline,
        retriever_cls,
        retriever_instance,
        validate_mock,
        embedding_calls,
        vector_store_calls,
        generator_calls,
    )


def test_factory_uses_exact_frozen_and_reranker_config_paths(tmp_path: Path) -> None:
    frozen_path = tmp_path / "frozen.json"
    reranker_path = tmp_path / "reranker.json"
    frozen_path.write_text(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    reranker_path.write_text(RERANKER_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    frozen_loader = MagicMock(side_effect=load_frozen_retrieval_config)
    reranker_loader = MagicMock(
        side_effect=__import__(
            "customer_claims_rag.retrieval.reranker",
            fromlist=["load_reranker_config"],
        ).load_reranker_config
    )
    settings = _application_settings(
        tmp_path,
        frozen_config_path=frozen_path,
        reranker_config_path=reranker_path,
    )

    _build_pipeline_with_recorders(
        tmp_path,
        settings=settings,
        frozen_config_loader=frozen_loader,
        reranker_config_loader=reranker_loader,
    )

    frozen_loader.assert_called_once_with(frozen_path)
    reranker_loader.assert_called_once_with(reranker_path)


def test_factory_propagates_frozen_config_loader_error(tmp_path: Path) -> None:
    def broken_loader(path: Path) -> FrozenRetrievalConfig:
        raise ValidationError.from_exception_data("FrozenRetrievalConfig", [])

    with pytest.raises(ValidationError):
        _build_pipeline_with_recorders(
            tmp_path,
            frozen_config_loader=broken_loader,
        )


def test_factory_propagates_reranker_config_loader_error(tmp_path: Path) -> None:
    def broken_loader(path: Path):
        raise FileNotFoundError(path)

    with pytest.raises(FileNotFoundError):
        _build_pipeline_with_recorders(
            tmp_path,
            reranker_config_loader=broken_loader,
        )


def test_factory_passes_retrieval_settings_to_embedding_and_vector_factories(
    tmp_path: Path,
) -> None:
    settings = _application_settings(tmp_path)
    _, _, _, _, embedding_calls, vector_store_calls, _ = _build_pipeline_with_recorders(
        tmp_path,
        settings=settings,
    )

    assert embedding_calls == [
        {
            "model_name": settings.retrieval.embedding_model,
            "api_key": settings.retrieval.openai_api_key,
        }
    ]
    assert vector_store_calls == [
        {
            "index_dir": settings.retrieval.index_dir,
            "collection_name": settings.retrieval.collection_name,
            "open_existing": True,
        }
    ]


def test_factory_builds_baseline_retriever_with_frozen_parameters(tmp_path: Path) -> None:
    frozen = _frozen_config()
    settings = _application_settings(tmp_path)
    _, retriever_cls, _, _, _, _, _ = _build_pipeline_with_recorders(tmp_path, settings=settings)

    retriever_cls.assert_called_once()
    kwargs = retriever_cls.call_args.kwargs
    assert kwargs["top_k"] == frozen.vector_top_k == 24
    assert kwargs["fetch_k"] == frozen.vector_fetch_k == 24
    assert kwargs["similarity_threshold"] == frozen.similarity_threshold == 0.0
    assert kwargs["index_dir"] == settings.retrieval.index_dir
    assert kwargs["embedding_provider"] is not None
    assert kwargs["vector_store"] is not None


def test_factory_does_not_use_env_default_top_k_for_baseline_retriever(tmp_path: Path) -> None:
    settings = _application_settings(tmp_path, retrieval=_retrieval_settings(tmp_path, top_k=4, fetch_k=12))
    _, retriever_cls, _, _, _, _, _ = _build_pipeline_with_recorders(tmp_path, settings=settings)
    kwargs = retriever_cls.call_args.kwargs
    assert kwargs["top_k"] == 24
    assert kwargs["fetch_k"] == 24
    assert kwargs["top_k"] != settings.retrieval.top_k
    assert kwargs["fetch_k"] != settings.retrieval.fetch_k


def test_factory_wires_frozen_retrieval_service_with_expected_provenance(
    tmp_path: Path,
) -> None:
    pipeline, _, _, _, _, _, _ = _build_pipeline_with_recorders(tmp_path)
    retrieval = pipeline._retrieval
    assert isinstance(retrieval, FrozenRetrievalService)
    assert retrieval.retrieval_config_id == "vector-pool-expansion-v1"
    assert retrieval.retrieval_config_version == "1.0.0"
    assert retrieval.reranker_config_id == "source-authority-v1"


def test_factory_validates_index_before_returning_pipeline(tmp_path: Path) -> None:
    validate_mock = MagicMock()
    _build_pipeline_with_recorders(tmp_path, validate_index=validate_mock)
    validate_mock.assert_called_once_with()


def test_factory_propagates_index_validation_error(tmp_path: Path) -> None:
    validate_mock = MagicMock(side_effect=IndexManifestError("manifest missing"))
    with pytest.raises(IndexManifestError, match="manifest missing"):
        _build_pipeline_with_recorders(tmp_path, validate_index=validate_mock)


def test_factory_does_not_return_pipeline_when_index_validation_fails(tmp_path: Path) -> None:
    diagnostics = ReleasePostureDiagnostics(
        release_posture_id="test",
        selected_target="active",
        target_status="selected_production_release",
        index_path_relative="index",
        corpus_fingerprint="test-fingerprint",
        chunk_count=1,
        document_count=1,
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        vector_dimension=8,
        frozen_retrieval_config_path="configs/retrieval/vector_pool_expansion_v1.json",
        frozen_config_hash="ff53ff9721ad86b1c542bf96dce616d9057ed3b347e341fed59750b07b69e048",
        vector_fetch_k=24,
        candidate_pool_k=24,
        final_top_k=12,
        similarity_threshold=0.0,
        reranker_id="source-authority-v1",
    )
    (tmp_path / "index").mkdir()
    with patch(
        "customer_claims_rag.application.factory.validate_release_posture_for_production",
        return_value=diagnostics,
    ), patch(
        "customer_claims_rag.application.factory.load_release_posture_descriptor",
        return_value=MagicMock(),
    ), patch(
        "customer_claims_rag.application.factory.BaselineRetriever"
    ) as retriever_cls:
        retriever_instance = MagicMock()
        retriever_instance.validate_index.side_effect = IndexManifestError("invalid index")
        retriever_cls.return_value = retriever_instance
        with pytest.raises(IndexManifestError):
            build_customer_claims_pipeline(
                _application_settings(tmp_path, index_dir=tmp_path / "index"),
                embedding_provider_factory=lambda **_: _RecordingEmbeddingProvider(),
                vector_store_factory=lambda **_: _RecordingVectorStore(),
                grounded_generator_factory=lambda _: MagicMock(spec=GroundedGenerator),
            )


def test_factory_passes_generation_settings_to_grounded_generator_factory(
    tmp_path: Path,
) -> None:
    settings = _application_settings(tmp_path)
    pipeline, _, _, _, _, _, generator_calls = _build_pipeline_with_recorders(
        tmp_path,
        settings=settings,
    )
    assert generator_calls == [settings.generation]
    assert isinstance(pipeline._generator, RiskAwareGroundedGenerator)


def test_factory_returns_customer_claims_pipeline(tmp_path: Path) -> None:
    pipeline, _, _, _, _, _, _ = _build_pipeline_with_recorders(tmp_path)
    assert isinstance(pipeline, CustomerClaimsPipeline)


def test_factory_propagates_embedding_factory_error(tmp_path: Path) -> None:
    def broken_embedding(**kwargs):
        raise EmbeddingError("embedding failed")

    with pytest.raises(EmbeddingError, match="embedding failed"):
        _build_pipeline_with_recorders(
            tmp_path,
            embedding_factory=broken_embedding,
        )


def test_factory_propagates_vector_store_factory_error(tmp_path: Path) -> None:
    def broken_store(**kwargs):
        raise RetrievalError("vector store failed")

    with pytest.raises(RetrievalError, match="vector store failed"):
        _build_pipeline_with_recorders(
            tmp_path,
            vector_store_factory=broken_store,
        )


def test_factory_propagates_grounded_generator_factory_error(tmp_path: Path) -> None:
    def broken_generator(settings: GenerationSettings) -> GroundedGenerator:
        raise GenerationError("generator failed")

    with pytest.raises(GenerationError, match="generator failed"):
        _build_pipeline_with_recorders(
            tmp_path,
            grounded_generator_factory=broken_generator,
        )


def test_factory_pipeline_handles_controlled_request(tmp_path: Path) -> None:
    from customer_claims_rag.generation.fallback import INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE
    from customer_claims_rag.generation.handoff import build_handoff_notice
    from customer_claims_rag.generation.models import GroundedGenerationResult
    from customer_claims_rag.generation.risk_integration_models import (
        RiskAwareGroundedGenerationResult,
    )
    from customer_claims_rag.risk import assess_deterministic_risk
    from customer_claims_rag.risk.models import RiskAssessmentRequest

    search_results = []
    retrieval = MagicMock()
    retrieval.search.return_value = search_results

    risk_assessment = assess_deterministic_risk(
        RiskAssessmentRequest(customer_query="test query"),
    )
    risk_result = RiskAwareGroundedGenerationResult(
        generation=GroundedGenerationResult(
            response_mode="insufficient_context",
            customer_response=INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
            citations=[],
        ),
        risk_assessment=risk_assessment,
        handoff_notice=build_handoff_notice(risk_assessment),
        generation_outcome="insufficient_context",
    )
    generator = MagicMock()
    generator.generate.return_value = risk_result

    pipeline = CustomerClaimsPipeline(retrieval=retrieval, generator=generator)
    result = pipeline.handle(CustomerClaimsRequest(customer_query="  test query  "))

    retrieval.search.assert_called_once_with("test query")
    generator.generate.assert_called_once()
    assert result.response is risk_result


def test_factory_does_not_create_openai_client_by_default_in_unit_wiring(tmp_path: Path) -> None:
    with patch(
        "customer_claims_rag.generation.factory.ChatOpenAI",
        side_effect=AssertionError("OpenAI client created"),
    ):
        _build_pipeline_with_recorders(tmp_path)


def _write_empty_index(tmp_path: Path, provider) -> Path:
    from customer_claims_rag.config import METADATA_SCHEMA_VERSION
    from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
    from customer_claims_rag.retrieval.manifest import build_manifest, write_manifest_atomic

    index_dir = tmp_path / "empty-index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    store.recreate_collection(embedding_dimension=provider.vector_dimension)
    assert store.count() == 0
    write_manifest_atomic(
        index_dir,
        build_manifest(
            collection_name="customer_claims",
            embedding_model=provider.model_name,
            corpus_fingerprint="empty-index",
            chunk_count=0,
            document_count=0,
            metadata_schema_version=METADATA_SCHEMA_VERSION,
            vector_dimension=provider.vector_dimension,
        ),
    )
    store.close()
    return index_dir


def test_factory_rejects_empty_schema_consistent_index_before_pipeline_return(
    tmp_path: Path,
) -> None:
    from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider

    provider = FakeEmbeddingProvider(model_name="fake-embedding-model", vector_dimension=8)
    index_dir = _write_empty_index(tmp_path, provider)
    descriptor_path = tmp_path / "release_descriptor.json"
    write_test_release_descriptor(
        descriptor_path,
        index_path_relative="empty-index",
        corpus_fingerprint="empty-index",
        chunk_count=1,
        document_count=1,
        supported_document_ids=["01_service_overview"],
    )
    settings = ApplicationSettings(
        retrieval=RetrievalSettings(
            index_dir=index_dir,
            collection_name="customer_claims",
            embedding_model="fake-embedding-model",
            top_k=4,
            fetch_k=12,
            similarity_threshold=0.0,
            embedding_batch_size=64,
            openai_api_key="dummy-test-key",
        ),
        generation=_generation_settings(),
        frozen_retrieval_config_path=FROZEN_CONFIG_PATH,
        reranker_config_path=RERANKER_CONFIG_PATH,
        release_target=resolved_release_target_for_index(
            index_dir,
            project_root=tmp_path,
            corpus_fingerprint="empty-index",
            chunk_count=1,
            document_count=1,
            supported_document_ids=("01_service_overview",),
        ),
        release_descriptor_path=descriptor_path,
    )
    generator_calls: list[GenerationSettings] = []

    def tracking_generator_factory(
        generation_settings: GenerationSettings,
    ) -> GroundedGenerator:
        generator_calls.append(generation_settings)
        return MagicMock(spec=GroundedGenerator)

    with pytest.raises(ReleasePostureError, match="chunk count"):
        build_customer_claims_pipeline(
            settings,
            embedding_provider_factory=lambda **_: provider,
            grounded_generator_factory=tracking_generator_factory,
        )

    assert generator_calls == []


@pytest.mark.parametrize(
    "module_name",
    [
        "customer_claims_rag.application.factory",
        "customer_claims_rag.application.settings",
    ],
)
def test_application_modules_do_not_import_evaluation(module_name: str) -> None:
    module = importlib.import_module(module_name)
    source_path = Path(module.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "customer_claims_rag.evaluation" not in alias.name
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "customer_claims_rag.evaluation" not in node.module


def test_settings_module_does_not_import_adapters() -> None:
    module = importlib.import_module("customer_claims_rag.application.settings")
    source_path = Path(module.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert ".adapters" not in alias.name
        if isinstance(node, ast.ImportFrom) and node.module:
            assert ".adapters" not in node.module


def test_factory_source_has_no_hardcoded_frozen_literals() -> None:
    source = (
        PROJECT_ROOT / "src" / "customer_claims_rag" / "application" / "factory.py"
    ).read_text(encoding="utf-8")
    assert "vector_top_k=24" not in source
    assert "final_top_k=12" not in source
    assert "candidate_pool_k=24" not in source


def test_factory_source_has_no_broad_exception_handler() -> None:
    source = (
        PROJECT_ROOT / "src" / "customer_claims_rag" / "application" / "factory.py"
    ).read_text(encoding="utf-8")
    assert "except Exception" not in source
