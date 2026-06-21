"""Git state helper tests."""

from __future__ import annotations

from customer_claims_rag.evaluation.git_state import read_git_state


def test_read_git_state_returns_commit(project_root) -> None:
    state = read_git_state(project_root)
    assert state.commit is not None
    assert state.dirty in {True, False}
    assert state.status_summary is not None


def test_read_git_state_unknown_when_git_unavailable(tmp_path, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr("customer_claims_rag.evaluation.git_state.subprocess.run", fail)
    state = read_git_state(tmp_path)
    assert state.commit is None
    assert state.dirty is None
