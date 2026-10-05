"""Serving a customer request never needs a tokenizer vocabulary.

``cl100k_base`` is needed where the canonical chunks are *built* and their topology verified. The
supported serving path (request validation, risk assessment, query embedding, Chroma retrieval,
reranking, context building, generation, customer-output projection) must not load it: a serving-only
image has no vocabulary cache, and a first request must not download one.

``tests/unit/test_embeddings.py`` already pins that the project's adapter switches LangChain's own
length check off, but with a recording stand-in for the LangChain client. Here the *real* LangChain
embedding and chat clients and the real OpenAI SDK run, and only the HTTP transport is replaced (no
socket is opened), so any tokenizer use hidden inside those libraries is executed and caught.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

import httpx
import pytest

from customer_claims_rag.application.customer_output import build_customer_output
from customer_claims_rag.application.frozen_retrieval import FrozenRetrievalService
from customer_claims_rag.application.models import CustomerClaimsRequest
from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
from customer_claims_rag.application.settings import load_frozen_retrieval_config
from customer_claims_rag.generation.adapters.openai_chat import OpenAIChatAdapter
from customer_claims_rag.generation.factory import build_chat_model, build_grounded_generator
from customer_claims_rag.generation.ports import ChatMessage
from customer_claims_rag.generation.risk_aware_generator import RiskAwareGroundedGenerator
from customer_claims_rag.generation_config import GenerationSettings
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.openai_embeddings import OpenAIEmbeddingProvider
from customer_claims_rag.retrieval.reranker import SourceAuthorityV1Reranker, load_reranker_config
from customer_claims_rag.retrieval.retriever import BaselineRetriever
from tests.release_posture_helpers import write_consistent_index
from tests.retrieval_helpers import make_chunk_record

ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSION = 8
QUERY_VECTOR = [1.0, 0.5, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0]
DUMMY_KEY = "sk-test-dummy-key-not-a-real-credential"


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def of(self, suffix: str) -> list[dict]:
        return [request for request in self.requests if request["url"].endswith(suffix)]


@pytest.fixture
def no_tokenizer(monkeypatch, tmp_path):
    """Every way of reaching a tokenizer vocabulary raises; both cache directories stay empty."""
    import tiktoken
    import tiktoken.load

    calls: list[str] = []

    def forbidden(name: str):
        def trap(*args, **kwargs):
            calls.append(f"{name}{args[:1]!r}")
            raise AssertionError(f"the serving path must not use a tokenizer: {name}")

        return trap

    monkeypatch.setattr(tiktoken, "get_encoding", forbidden("get_encoding"))
    monkeypatch.setattr(tiktoken, "encoding_for_model", forbidden("encoding_for_model"))
    monkeypatch.setattr(tiktoken.load, "read_file", forbidden("read_file"))
    monkeypatch.setattr(tiktoken.load, "read_file_cached", forbidden("read_file_cached"))
    tiktoken_cache = tmp_path / "empty-tiktoken-cache"
    data_gym_cache = tmp_path / "empty-data-gym-cache"
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tiktoken_cache))
    monkeypatch.setenv("DATA_GYM_CACHE_DIR", str(data_gym_cache))
    yield calls
    assert calls == []
    assert not tiktoken_cache.exists()
    assert not data_gym_cache.exists()


@pytest.fixture
def openai_http(monkeypatch, no_tokenizer) -> _Recorder:
    """The OpenAI HTTP API, answered in-process: no socket, real LangChain + OpenAI SDK above it."""
    recorder = _Recorder()
    draft = {"response_mode": "grounded_answer", "answer": "Сроки доставки указаны в правилах. [S1]"}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url).split("?")[0]
        body = json.loads(request.content.decode("utf-8"))
        recorder.requests.append({"url": url, "body": body})
        if url.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            if body.get("encoding_format") == "base64":
                payload = base64.b64encode(struct.pack(f"<{DIMENSION}f", *QUERY_VECTOR)).decode()
            else:
                payload = QUERY_VECTOR
            return httpx.Response(
                200,
                request=request,
                json={
                    "object": "list",
                    "model": EMBEDDING_MODEL,
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    "data": [
                        {"object": "embedding", "index": index, "embedding": payload}
                        for index in range(len(inputs))
                    ],
                },
            )
        if url.endswith("/chat/completions"):
            return httpx.Response(
                200,
                request=request,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(draft, ensure_ascii=False)},
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        return httpx.Response(404, request=request, json={"error": f"unexpected url {url}"})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    return recorder


def _generation_settings() -> GenerationSettings:
    return GenerationSettings(
        model_name="gpt-4o-mini",
        temperature=0.0,
        timeout_seconds=30.0,
        max_retries=0,
        max_output_tokens=256,
        prompt_path=(ROOT / "prompts" / "system_prompt.md").resolve(),
    ).validate()


def test_langchain_embedding_requests_carry_plain_text_and_need_no_tokenizer(openai_http) -> None:
    provider = OpenAIEmbeddingProvider(model_name=EMBEDDING_MODEL, api_key=DUMMY_KEY)

    assert provider.embed_query("Где мой заказ?") == pytest.approx(QUERY_VECTOR)
    assert len(provider.embed_documents(["первый фрагмент", "второй фрагмент"])) == 2

    sent = openai_http.of("/embeddings")
    assert len(sent) == 2
    for request in sent:
        inputs = request["body"]["input"] if isinstance(request["body"]["input"], list) else [request["body"]["input"]]
        # token ids (lists of ints) would mean LangChain tokenized the text itself
        assert inputs and all(isinstance(item, str) for item in inputs)


def test_chat_completion_needs_no_tokenizer(openai_http) -> None:
    adapter = build_chat_model(_generation_settings())
    assert isinstance(adapter, OpenAIChatAdapter)

    text = adapter.complete(
        [ChatMessage(role="system", content="Отвечай кратко."), ChatMessage(role="user", content="Привет")]
    )

    assert "grounded_answer" in text
    assert len(openai_http.of("/chat/completions")) == 1


def _pipeline(tmp_path: Path) -> CustomerClaimsPipeline:
    chunks = [
        make_chunk_record(
            chunk_id=f"02_delivery_rules::chunk-{index:03d}",
            document_id="02_delivery_rules",
            source_path="data/02_clean_markdown/02_delivery_rules.md",
            heading=f"Правило {index}",
            content=f"Документ: доставка | Раздел: Правило {index}\n---\nТекст правила доставки номер {index}.",
            chunk_index=index,
        )
        for index in range(1, 6)
    ]
    index_dir = tmp_path / "index"
    write_consistent_index(index_dir, chunks, embedding_model=EMBEDDING_MODEL, vector_dimension=DIMENSION)
    frozen = load_frozen_retrieval_config(ROOT / "configs" / "retrieval" / "vector_pool_expansion_v1.json")
    retriever = BaselineRetriever(
        embedding_provider=OpenAIEmbeddingProvider(model_name=EMBEDDING_MODEL, api_key=DUMMY_KEY),
        vector_store=ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims", open_existing=True),
        index_dir=index_dir,
        top_k=frozen.vector_top_k,
        fetch_k=frozen.vector_fetch_k,
        similarity_threshold=frozen.similarity_threshold,
    )
    retriever.validate_index()
    reranker = SourceAuthorityV1Reranker(load_reranker_config(ROOT / "configs" / "reranking" / "source_authority_v1.json"))
    return CustomerClaimsPipeline(
        retrieval=FrozenRetrievalService(retriever=retriever, reranker=reranker, config=frozen),
        generator=RiskAwareGroundedGenerator(grounded_generator=build_grounded_generator(_generation_settings())),
    )


def test_a_full_customer_request_needs_no_tokenizer(openai_http, tmp_path) -> None:
    pipeline = _pipeline(tmp_path)

    ordinary = build_customer_output(
        pipeline.handle(CustomerClaimsRequest(customer_query="Сколько обычно идет доставка?"))
    )
    critical = build_customer_output(
        pipeline.handle(CustomerClaimsRequest(customer_query="После еды стало трудно дышать, состояние ухудшается."))
    )

    # the whole path ran: retrieval reached the model through the real clients ...
    assert ordinary.provenance == "llm_draft"
    assert [source.document_id for source in ordinary.answer_sources] == ["02_delivery_rules"]
    # ... and the deterministic safety text replaced the model draft for the critical request
    assert critical.provenance == "category_template"
    assert critical.priority_handoff is True
    assert len(openai_http.of("/embeddings")) == 2
    assert len(openai_http.of("/chat/completions")) == 2
