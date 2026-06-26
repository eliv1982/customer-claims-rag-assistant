"""Integration tests for production pipeline wiring with temporary Chroma."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from customer_claims_rag.application.factory import build_customer_claims_pipeline
from customer_claims_rag.application.frozen_retrieval import FrozenRetrievalService
from customer_claims_rag.application.models import CustomerClaimsRequest
from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
from customer_claims_rag.application.settings import ApplicationSettings, load_frozen_retrieval_config
from customer_claims_rag.release.posture import ReleasePostureDiagnostics
from customer_claims_rag.config import METADATA_SCHEMA_VERSION
from customer_claims_rag.exceptions import IndexManifestError, LLMCallError, ReleasePostureError
from customer_claims_rag.generation.adapters.fake_chat import FakeChatModel
from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
)
from customer_claims_rag.generation.generator import GroundedGenerator
from customer_claims_rag.generation.handoff import HIGH_HANDOFF_NOTICE
from customer_claims_rag.generation.prompt_builder import PromptBuilder
from customer_claims_rag.generation.risk_aware_generator import RiskAwareGroundedGenerator
from customer_claims_rag.generation_config import GenerationSettings
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval.manifest import build_manifest, write_manifest_atomic
from customer_claims_rag.retrieval.retriever import BaselineRetriever
from customer_claims_rag.retrieval_config import RetrievalSettings
from tests.release_posture_helpers import (
    resolved_release_target_for_index,
    write_test_release_descriptor,
)
from customer_claims_rag.risk.models import RiskLevel
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from tests.retrieval_helpers import make_chunk_record

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "grounded_answer_v1.md"
FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval" / "vector_pool_expansion_v1.json"
RERANKER_CONFIG_PATH = PROJECT_ROOT / "configs" / "reranking" / "source_authority_v1.json"
EMBEDDING_MODEL = "fake-embedding-model"


def _generation_settings() -> GenerationSettings:
    return GenerationSettings(
        model_name="gpt-4o-mini",
        temperature=0.0,
        timeout_seconds=60.0,
        max_retries=2,
        max_output_tokens=1024,
        prompt_path=PROMPT_PATH.resolve(),
    ).validate()


def _write_index(
    tmp_path: Path,
    provider: FakeEmbeddingProvider,
    chunks,
) -> Path:
    index_dir = tmp_path / "index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    vectors = provider.embed_documents([chunk.content for chunk in chunks])
    store.recreate_collection(embedding_dimension=provider.vector_dimension)
    store.add_chunks(chunks, vectors)
    manifest = build_manifest(
        collection_name="customer_claims",
        embedding_model=provider.model_name,
        corpus_fingerprint="integration-test-fingerprint",
        chunk_count=len(chunks),
        document_count=len({chunk.document_id for chunk in chunks}),
        metadata_schema_version=METADATA_SCHEMA_VERSION,
        vector_dimension=provider.vector_dimension,
    )
    write_manifest_atomic(index_dir, manifest)
    store.close()
    return index_dir


def _write_empty_index(
    tmp_path: Path,
    provider: FakeEmbeddingProvider,
) -> Path:
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


def _release_params_from_index(index_dir: Path) -> tuple[int, tuple[str, ...], str]:
    from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
    from customer_claims_rag.retrieval.manifest import load_manifest

    manifest = load_manifest(index_dir)
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    document_ids = tuple(sorted(store.list_document_ids()))
    store.close()
    return manifest.chunk_count, document_ids, manifest.corpus_fingerprint


def _application_settings(
    tmp_path: Path,
    index_dir: Path,
    *,
    frozen_config_path: Path = FROZEN_CONFIG_PATH,
    corpus_fingerprint: str | None = None,
    chunk_count: int | None = None,
    document_ids: tuple[str, ...] | None = None,
    expected_frozen_config_hash: str | None = None,
) -> ApplicationSettings:
    if chunk_count is None or document_ids is None or corpus_fingerprint is None:
        loaded_chunk_count, loaded_document_ids, loaded_fingerprint = _release_params_from_index(
            index_dir
        )
        chunk_count = chunk_count if chunk_count is not None else loaded_chunk_count
        document_ids = document_ids if document_ids is not None else loaded_document_ids
        corpus_fingerprint = (
            corpus_fingerprint if corpus_fingerprint is not None else loaded_fingerprint
        )
    if expected_frozen_config_hash is None:
        from customer_claims_rag.evaluation.pool_expansion_metrics import (
            compute_pool_expansion_config_hash,
            load_pool_expansion_config,
        )

        expected_frozen_config_hash = compute_pool_expansion_config_hash(
            load_pool_expansion_config(frozen_config_path)
        )
    descriptor_path = tmp_path / "release_descriptor.json"
    relative_index = index_dir.resolve().relative_to(tmp_path.resolve()).as_posix()
    write_test_release_descriptor(
        descriptor_path,
        index_path_relative=relative_index,
        corpus_fingerprint=corpus_fingerprint,
        chunk_count=chunk_count,
        document_count=len(document_ids),
        supported_document_ids=list(document_ids),
        expected_frozen_config_hash=expected_frozen_config_hash,
    )
    return ApplicationSettings(
        retrieval=RetrievalSettings(
            index_dir=index_dir,
            collection_name="customer_claims",
            embedding_model=EMBEDDING_MODEL,
            top_k=4,
            fetch_k=12,
            similarity_threshold=0.0,
            embedding_batch_size=64,
            openai_api_key="dummy-not-real-key",
        ),
        generation=_generation_settings(),
        frozen_retrieval_config_path=frozen_config_path,
        reranker_config_path=RERANKER_CONFIG_PATH,
        release_target=resolved_release_target_for_index(
            index_dir,
            project_root=tmp_path,
            corpus_fingerprint=corpus_fingerprint,
            chunk_count=chunk_count,
            document_count=len(document_ids),
            supported_document_ids=document_ids,
            embedding_model=EMBEDDING_MODEL,
        ),
        release_descriptor_path=descriptor_path,
    )


def _embedding_factory(provider: FakeEmbeddingProvider):
    def factory(**kwargs):
        return provider

    return factory


def _grounded_generator_factory(chat_model: FakeChatModel):
    def factory(settings: GenerationSettings) -> GroundedGenerator:
        return GroundedGenerator(
            chat_model=chat_model,
            prompt_builder=PromptBuilder(prompt_path=settings.prompt_path),
        )

    return factory


def _release_validation_patches():
    diagnostics = ReleasePostureDiagnostics(
        release_posture_id="test",
        selected_target="active",
        target_status="selected_production_release",
        index_path_relative="index",
        corpus_fingerprint="integration-test-fingerprint",
        chunk_count=1,
        document_count=1,
        collection_name="customer_claims",
        embedding_model=EMBEDDING_MODEL,
        vector_dimension=8,
        frozen_retrieval_config_path="configs/retrieval/vector_pool_expansion_v1.json",
        frozen_config_hash="ff53ff9721ad86b1c542bf96dce616d9057ed3b347e341fed59750b07b69e048",
        vector_fetch_k=24,
        candidate_pool_k=24,
        final_top_k=12,
        similarity_threshold=0.0,
        reranker_id="source-authority-v1",
    )
    return (
        patch(
            "customer_claims_rag.application.factory.validate_release_posture_for_production",
            return_value=diagnostics,
        ),
        patch(
            "customer_claims_rag.application.factory.load_release_posture_descriptor",
            return_value=MagicMock(),
        ),
    )


def _build_pipeline(
    settings: ApplicationSettings,
    provider: FakeEmbeddingProvider,
    chat_model: FakeChatModel,
    *,
    bypass_release_validation: bool = False,
) -> CustomerClaimsPipeline:
    if bypass_release_validation:
        validation_patches = _release_validation_patches()
        with validation_patches[0], validation_patches[1]:
            return build_customer_claims_pipeline(
                settings,
                embedding_provider_factory=_embedding_factory(provider),
                grounded_generator_factory=_grounded_generator_factory(chat_model),
            )
    return build_customer_claims_pipeline(
        settings,
        embedding_provider_factory=_embedding_factory(provider),
        grounded_generator_factory=_grounded_generator_factory(chat_model),
    )


def _dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _find_source_authority_fixture(
    provider: FakeEmbeddingProvider,
) -> tuple[str, str, str]:
    for index in range(1000):
        query = f"source-authority-query-{index}"
        faq_content = f"faq-source-authority-{index}"
        policy_content = f"policy-source-authority-{index}"
        query_vector = provider.embed_query(query)
        faq_vector = provider.embed_documents([faq_content])[0]
        policy_vector = provider.embed_documents([policy_content])[0]
        faq_similarity = _dot_product(query_vector, faq_vector)
        policy_similarity = _dot_product(query_vector, policy_vector)
        gap = faq_similarity - policy_similarity
        if 0.0 < gap < 0.03:
            return query, faq_content, policy_content
    pytest.fail("unable to build deterministic source-authority fixture")


def _packaging_chunks():
    return [
        make_chunk_record(
            chunk_id="06_food_quality_and_packaging::1",
            document_id="06_food_quality_and_packaging",
            content=(
                "Документ: Качество и упаковка | Раздел: Упаковка\n---\n"
                "Правила фиксации вскрытой упаковки и качества продукции."
            ),
            strategy="policy",
            heading="Упаковка",
            heading_path=["Упаковка"],
            topic="packaging",
            risk_level="high",
        ),
    ]


def _valid_grounded_json(answer: str) -> str:
    escaped = answer.replace('"', '\\"')
    return f'{{"response_mode":"grounded_answer","answer":"{escaped}"}}'


def test_end_to_end_success_with_temporary_chroma(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(model_name=EMBEDDING_MODEL, vector_dimension=8)
    chunks = _packaging_chunks()
    index_dir = _write_index(tmp_path, provider, chunks)
    settings = _application_settings(tmp_path, index_dir)
    chat_model = FakeChatModel(
        response=_valid_grounded_json("По правилам вскрытая упаковка фиксируется [S1]."),
    )
    pipeline = _build_pipeline(settings, provider, chat_model)

    assert isinstance(pipeline, CustomerClaimsPipeline)
    assert isinstance(pipeline._retrieval, FrozenRetrievalService)
    assert isinstance(pipeline._generator, RiskAwareGroundedGenerator)
    assert pipeline._retrieval.retrieval_config_id == "vector-pool-expansion-v1"
    assert pipeline._retrieval.reranker_config_id == "source-authority-v1"

    result = pipeline.handle(
        CustomerClaimsRequest(customer_query="  Упаковка была вскрыта.  "),
    )

    assert result.response.generation_outcome == "grounded_answer"
    assert result.response.generation.response_mode == "grounded_answer"
    assert "[S1]" in result.response.generation.customer_response
    assert len(result.response.generation.citations) == 1
    assert (
        result.response.generation.citations[0].document_id
        == "06_food_quality_and_packaging"
    )
    assert result.response.risk_assessment.risk_floor is RiskLevel.HIGH
    assert result.response.handoff_notice == HIGH_HANDOFF_NOTICE
    assert RiskReasonCode.PACKAGE_TAMPERING in result.response.risk_assessment.reason_codes
    assert chat_model.call_count == 1


def test_baseline_retriever_created_with_frozen_parameters(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(model_name=EMBEDDING_MODEL, vector_dimension=8)
    index_dir = _write_index(tmp_path, provider, _packaging_chunks())
    settings = _application_settings(tmp_path, index_dir)
    frozen = load_frozen_retrieval_config(settings.frozen_retrieval_config_path)

    pipeline = _build_pipeline(
        settings,
        provider,
        FakeChatModel(response=_valid_grounded_json("Ответ [S1].")),
    )
    retriever = pipeline._retrieval._retriever
    assert isinstance(retriever, BaselineRetriever)
    assert retriever.top_k == frozen.vector_top_k == 24
    assert retriever.fetch_k == frozen.vector_fetch_k == 24
    assert retriever.similarity_threshold == frozen.similarity_threshold == 0.0


def test_source_authority_reranking_prefers_policy_over_faq(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(model_name=EMBEDDING_MODEL, vector_dimension=8)
    query, faq_content, policy_content = _find_source_authority_fixture(provider)
    faq_chunk = make_chunk_record(
        chunk_id="faq::1",
        document_id="faq_doc",
        content=faq_content,
        strategy="faq",
        heading="FAQ",
    )
    policy_chunk = make_chunk_record(
        chunk_id="policy::1",
        document_id="policy_doc",
        content=policy_content,
        strategy="policy",
        heading="Policy",
    )
    index_dir = _write_index(tmp_path, provider, [faq_chunk, policy_chunk])
    settings = _application_settings(tmp_path, index_dir)

    baseline = BaselineRetriever(
        embedding_provider=provider,
        vector_store=ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims"),
        index_dir=index_dir,
        top_k=24,
        fetch_k=24,
        similarity_threshold=0.0,
    )
    baseline_response = baseline.search(query)
    assert baseline_response.results[0].chunk_type == "faq"

    pipeline = _build_pipeline(
        settings,
        provider,
        FakeChatModel(response=_valid_grounded_json("Ответ [S1].")),
    )
    final_results = pipeline._retrieval.search(query)
    assert final_results[0].chunk_type == "policy"
    assert final_results[0].chunk_id == "policy::1"


def test_empty_retrieval_returns_insufficient_context_without_chat_call(
    tmp_path: Path,
) -> None:
    provider = FakeEmbeddingProvider(model_name=EMBEDDING_MODEL, vector_dimension=8)
    index_dir = _write_index(tmp_path, provider, _packaging_chunks())

    frozen_payload = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    frozen_payload["threshold"] = 1.0
    frozen_path = tmp_path / "frozen-high-threshold.json"
    frozen_path.write_text(json.dumps(frozen_payload), encoding="utf-8")

    settings = _application_settings(tmp_path, index_dir, frozen_config_path=frozen_path)
    chat_model = FakeChatModel(
        response=_valid_grounded_json("Не должно вызываться."),
    )
    pipeline = _build_pipeline(
        settings,
        provider,
        chat_model,
        bypass_release_validation=True,
    )

    result = pipeline.handle(
        CustomerClaimsRequest(customer_query="Упаковка была вскрыта."),
    )

    assert chat_model.call_count == 0
    assert result.response.generation_outcome == "insufficient_context"
    assert result.response.generation.response_mode == "insufficient_context"
    assert result.response.generation.customer_response == INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE
    assert result.response.generation.citations == []
    assert result.response.risk_assessment.risk_floor is RiskLevel.HIGH
    assert result.response.handoff_notice == HIGH_HANDOFF_NOTICE


def test_generation_fallback_preserves_risk_for_high_risk_query(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(model_name=EMBEDDING_MODEL, vector_dimension=8)
    index_dir = _write_index(tmp_path, provider, _packaging_chunks())
    settings = _application_settings(tmp_path, index_dir)
    chat_model = FakeChatModel(response="{bad-json")
    pipeline = _build_pipeline(settings, provider, chat_model)

    result = pipeline.handle(
        CustomerClaimsRequest(customer_query="Упаковка была вскрыта."),
    )

    assert result.response.generation_outcome == "generation_error_fallback"
    assert result.response.generation.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE
    assert result.response.generation.citations == []
    assert result.response.risk_assessment.risk_floor is RiskLevel.HIGH
    assert result.response.handoff_notice == HIGH_HANDOFF_NOTICE


def test_generation_llm_call_error_fallback_preserves_risk(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(model_name=EMBEDDING_MODEL, vector_dimension=8)
    index_dir = _write_index(tmp_path, provider, _packaging_chunks())
    settings = _application_settings(tmp_path, index_dir)
    chat_model = FakeChatModel(error=LLMCallError("boom"))
    pipeline = _build_pipeline(settings, provider, chat_model)

    result = pipeline.handle(
        CustomerClaimsRequest(customer_query="Упаковка была вскрыта."),
    )

    assert result.response.generation_outcome == "generation_error_fallback"
    assert result.response.risk_assessment.risk_floor is RiskLevel.HIGH
    assert result.response.handoff_notice == HIGH_HANDOFF_NOTICE


def _manual_application_settings(
    tmp_path: Path,
    index_dir: Path,
    *,
    corpus_fingerprint: str,
    chunk_count: int,
    document_ids: tuple[str, ...],
    frozen_config_path: Path = FROZEN_CONFIG_PATH,
) -> ApplicationSettings:
    descriptor_path = tmp_path / "release_descriptor.json"
    relative_index = (
        index_dir.resolve().relative_to(tmp_path.resolve()).as_posix()
        if index_dir.exists()
        else index_dir.name
    )
    write_test_release_descriptor(
        descriptor_path,
        index_path_relative=relative_index,
        corpus_fingerprint=corpus_fingerprint,
        chunk_count=chunk_count,
        document_count=len(document_ids),
        supported_document_ids=list(document_ids),
    )
    return ApplicationSettings(
        retrieval=RetrievalSettings(
            index_dir=index_dir,
            collection_name="customer_claims",
            embedding_model=EMBEDDING_MODEL,
            top_k=4,
            fetch_k=12,
            similarity_threshold=0.0,
            embedding_batch_size=64,
            openai_api_key="dummy-not-real-key",
        ),
        generation=_generation_settings(),
        frozen_retrieval_config_path=frozen_config_path,
        reranker_config_path=RERANKER_CONFIG_PATH,
        release_target=resolved_release_target_for_index(
            index_dir,
            project_root=tmp_path,
            corpus_fingerprint=corpus_fingerprint,
            chunk_count=chunk_count,
            document_count=len(document_ids),
            supported_document_ids=document_ids,
            embedding_model=EMBEDDING_MODEL,
        ),
        release_descriptor_path=descriptor_path,
    )


def test_missing_index_fails_fast_without_building_pipeline(tmp_path: Path) -> None:
    missing_index = tmp_path / "missing-index"
    settings = _manual_application_settings(
        tmp_path,
        missing_index,
        corpus_fingerprint="missing",
        chunk_count=1,
        document_ids=("01_service_overview",),
    )
    provider = FakeEmbeddingProvider(model_name=EMBEDDING_MODEL, vector_dimension=8)

    with pytest.raises(ReleasePostureError, match="does not exist"):
        build_customer_claims_pipeline(
            settings,
            embedding_provider_factory=_embedding_factory(provider),
            grounded_generator_factory=_grounded_generator_factory(FakeChatModel()),
        )


def test_empty_chroma_index_rejected_before_pipeline_return(
    tmp_path: Path,
    project_root: Path,
) -> None:
    provider = FakeEmbeddingProvider(model_name=EMBEDDING_MODEL, vector_dimension=8)
    index_dir = _write_empty_index(tmp_path, provider)
    settings = _manual_application_settings(
        tmp_path,
        index_dir,
        corpus_fingerprint="empty-index",
        chunk_count=1,
        document_ids=("01_service_overview",),
    )
    production_index = project_root / "data" / "04_index"
    before = {path.name for path in production_index.iterdir()} if production_index.is_dir() else set()

    with pytest.raises(ReleasePostureError, match="chunk count"):
        build_customer_claims_pipeline(
            settings,
            embedding_provider_factory=_embedding_factory(provider),
            grounded_generator_factory=_grounded_generator_factory(FakeChatModel()),
        )

    after = {path.name for path in production_index.iterdir()} if production_index.is_dir() else set()
    assert after == before
    assert all(path.is_relative_to(tmp_path) for path in index_dir.rglob("*") if path.is_file() or path.is_dir())


def test_missing_index_does_not_mutate_production_index(
    tmp_path: Path,
    project_root: Path,
) -> None:
    production_index = project_root / "data" / "04_index"
    before = {path.name for path in production_index.iterdir()} if production_index.is_dir() else set()

    settings = _manual_application_settings(
        tmp_path,
        tmp_path / "no-index",
        corpus_fingerprint="missing",
        chunk_count=1,
        document_ids=("01_service_overview",),
    )
    provider = FakeEmbeddingProvider(model_name=EMBEDDING_MODEL, vector_dimension=8)

    with pytest.raises(ReleasePostureError):
        build_customer_claims_pipeline(
            settings,
            embedding_provider_factory=_embedding_factory(provider),
            grounded_generator_factory=_grounded_generator_factory(FakeChatModel()),
        )

    after = {path.name for path in production_index.iterdir()} if production_index.is_dir() else set()
    assert after == before
