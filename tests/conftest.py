"""Shared pytest fixtures and the hermetic-test contract.

The default suite (``python -m pytest`` or ``pytest``) must pass on a fresh clone
without an OpenAI key, a ``.env`` file, a tiktoken cache, built indexes or any
network access, and must not write into the repository:

* app-related environment variables are removed for every test, and the
  maintainer's real ``.env`` is never read (``dummy_openai_api_key`` provides a
  fake key where a test needs one);
* ``tiktoken`` resolves to a deterministic local encoding (``tests/offline_tiktoken.py``),
  which is NOT cl100k_base: this lane does not verify production tokenization. Tests that
  need real cl100k values carry ``@pytest.mark.real_tiktoken`` and run in the separate real
  lane (``tests/tokenizer_lanes.py``), which needs a provisioned tiktoken cache and refuses
  to run, rather than skip or download, without it;
* non-loopback network access in pytest and ordinary child Python interpreters raises and fails the
  session (``tests/network_guard.py`` plus the inherited test-only ``sitecustomize`` hook);
* tests that verify gitignored local artifacts carry ``local_artifact`` and skip
  with a reason when the artifact is absent (``tests/local_artifacts.py``).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tests.network_guard import NetworkGuard, PythonSubprocessNetworkGuard
from tests.offline_tiktoken import offline_tiktoken
from tests.tokenizer_lanes import (
    DEFAULT_LANE_NOTE,
    REAL_LANE_COMMAND,
    REAL_LANE_NOTE,
    SKIP_REASON_DEFAULT_LANE,
    RealTokenizerNotProvisioned,
    require_provisioned_cl100k,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_MARKDOWN_DIR = PROJECT_ROOT / "data" / "02_clean_markdown"
# Frozen copy of the 15 documents as they were when the retrieval experiments ran
# (experiments/corpus/historical_pre_2d2_15doc_v1/snapshot_manifest.json). The live corpus
# has moved on; evidence about the old one is checked against this copy.
HISTORICAL_CORPUS_DIR = PROJECT_ROOT / "experiments" / "corpus" / "historical_pre_2d2_15doc_v1"

DUMMY_OPENAI_API_KEY = "sk-test-dummy-key-not-a-real-credential"
_APP_ENV_PREFIXES = ("OPENAI_", "RAG_", "GENERATION_")
_APP_ENV_NAMES = ("CUSTOMER_CLAIMS_PROJECT_ROOT",)

_NETWORK_GUARD = NetworkGuard()
_PYTHON_SUBPROCESS_NETWORK_GUARD = PythonSubprocessNetworkGuard()


def _network_attempts() -> list[str]:
    return [*_NETWORK_GUARD.attempts, *_PYTHON_SUBPROCESS_NETWORK_GUARD.attempts]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--real-tiktoken",
        action="store_true",
        default=False,
        help=(
            "real-tokenizer lane: use the real cl100k_base vocabulary instead of the offline "
            "stand-in and run real_tiktoken tests. The vocabulary must already be in the "
            "local tiktoken cache; pytest stops at setup if it is not (nothing is downloaded). "
            f"Use: {REAL_LANE_COMMAND}"
        ),
    )


def _real_lane(config: pytest.Config) -> bool:
    return bool(config.getoption("--real-tiktoken"))


def pytest_configure(config: pytest.Config) -> None:
    _NETWORK_GUARD.install()
    _PYTHON_SUBPROCESS_NETWORK_GUARD.install()
    if _real_lane(config):
        try:
            require_provisioned_cl100k()
        except RealTokenizerNotProvisioned as exc:
            raise pytest.UsageError(str(exc)) from None


def pytest_unconfigure(config: pytest.Config) -> None:
    _PYTHON_SUBPROCESS_NETWORK_GUARD.uninstall()
    _NETWORK_GUARD.uninstall()


def pytest_report_header(config: pytest.Config) -> list[str]:
    return [REAL_LANE_NOTE if _real_lane(config) else DEFAULT_LANE_NOTE]


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _real_lane(config):
        return
    real_items = [item for item in items if "real_tiktoken" in item.keywords]
    if real_items and len(real_items) == len(items) and not config.option.collectonly:
        # e.g. ``-m real_tiktoken`` without the flag: every test would skip and the run
        # would look green although nothing about production tokenization was checked.
        pytest.exit(
            "only real_tiktoken tests were selected, but the real-tokenizer lane was not "
            f"requested; run: {REAL_LANE_COMMAND}",
            returncode=int(pytest.ExitCode.USAGE_ERROR),
        )
    skip = pytest.mark.skip(reason=SKIP_REASON_DEFAULT_LANE)
    for item in real_items:
        item.add_marker(skip)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and _real_lane(item.config) and "real_tiktoken" in item.keywords:
        # The real lane exists to run these tests; a skip there is a gap, not a result.
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        report.outcome = "failed"
        report.longrepr = f"real_tiktoken test skipped in the real-tokenizer lane: {reason}"


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    config = terminalreporter.config
    if config.get_verbosity() < 0:
        # ``-q`` suppresses the header, so the lane note is repeated where it stays visible.
        terminalreporter.write_line(REAL_LANE_NOTE if _real_lane(config) else DEFAULT_LANE_NOTE)
    attempts = _network_attempts()
    if attempts:
        terminalreporter.section("blocked network access", red=True)
        for attempt in attempts:
            terminalreporter.write_line(attempt)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # An attempt that code under test swallowed must still fail the run.
    if _network_attempts() and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(scope="session", autouse=True)
def _tokenizer_mode(request: pytest.FixtureRequest):
    if _real_lane(request.config):
        yield
        return
    with offline_tiktoken():
        yield


@pytest.fixture(autouse=True)
def _hermetic_app_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No developer shell state and no maintainer ``.env`` leaks into a test."""
    from customer_claims_rag import env_bootstrap

    for name in list(os.environ):
        if name.startswith(_APP_ENV_PREFIXES) or name in _APP_ENV_NAMES:
            monkeypatch.delenv(name, raising=False)

    real_load_dotenv = env_bootstrap.load_dotenv

    def load_dotenv(dotenv_path=None, *args, **kwargs):
        if dotenv_path is not None and Path(dotenv_path).resolve().parent == PROJECT_ROOT:
            return False
        return real_load_dotenv(dotenv_path, *args, **kwargs)

    monkeypatch.setattr(env_bootstrap, "load_dotenv", load_dotenv)


@pytest.fixture
def dummy_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """A syntactically plausible fake key for code paths that only require one to exist."""
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_OPENAI_API_KEY)
    return DUMMY_OPENAI_API_KEY


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def clean_markdown_dir() -> Path:
    return CLEAN_MARKDOWN_DIR


@pytest.fixture(scope="session")
def corpus_sandbox(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Throwaway project root with the clean corpus and experiment overlays.

    Corpus-overlay builders stage files under ``<permitted_root>/.tmp``; building
    against the real repository root would leave that directory in the checkout.
    """
    root = tmp_path_factory.mktemp("corpus_sandbox")
    shutil.copytree(CLEAN_MARKDOWN_DIR, root / "data" / "02_clean_markdown")
    shutil.copytree(PROJECT_ROOT / "experiments" / "corpus", root / "experiments" / "corpus")
    return root


@pytest.fixture(scope="session")
def historical_corpus_sandbox(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``corpus_sandbox`` with the frozen historical corpus in place of the live one.

    The sandbox keeps the logical layout (``data/02_clean_markdown/<id>.md``), so chunk source
    paths, and therefore every fingerprint, are the ones the tracked experiment artifacts record.
    """
    root = tmp_path_factory.mktemp("historical_corpus_sandbox")
    clean = root / "data" / "02_clean_markdown"
    clean.mkdir(parents=True)
    for source in sorted(HISTORICAL_CORPUS_DIR.glob("*.md")):
        shutil.copy(source, clean / source.name)
    shutil.copytree(PROJECT_ROOT / "experiments" / "corpus", root / "experiments" / "corpus")
    return root


@pytest.fixture
def temp_project(tmp_path: Path, clean_markdown_dir: Path) -> Path:
    """Isolated project root with a copy of clean Markdown corpus."""
    data_dir = tmp_path / "data"
    clean_dest = data_dir / "02_clean_markdown"
    clean_dest.mkdir(parents=True)
    for source in sorted(clean_markdown_dir.glob("*.md")):
        shutil.copy(source, clean_dest / source.name)
    (data_dir / "03_chunks").mkdir()
    (data_dir / "04_index").mkdir()
    return tmp_path


@pytest.fixture
def builder(temp_project: Path) -> "CorpusBuilder":
    from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
    from customer_claims_rag.token_counter import TiktokenCounter

    return CorpusBuilder(
        token_counter=TiktokenCounter(),
        permitted_root=temp_project.resolve(),
    )


@pytest.fixture
def fake_embedding_provider():
    from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider

    return FakeEmbeddingProvider(model_name="fake-embedding-model", vector_dimension=8)


@pytest.fixture
def sample_metadata() -> dict:
    return {
        "document_id": "99_test_doc",
        "title": "Тестовый документ",
        "category": "general",
        "document_type": "reference",
        "version": "1.0.0",
        "status": "active",
        "effective_date": "2026-06-01",
        "last_updated": "2026-06-20",
        "audience": "support",
        "confidentiality": "internal",
        "source_type": "internal_reference",
        "language": "ru",
        "priority": "medium",
    }
