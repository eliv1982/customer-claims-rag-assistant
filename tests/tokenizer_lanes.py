"""The two tokenizer lanes of the test suite, and what each one does and does not prove.

default lane      ``python -m pytest``
    ``tiktoken`` resolves to ``tests/offline_tiktoken.py``, a deterministic word-level
    stand-in. Nothing needs the network or a tokenizer cache, but chunk packing is NOT the
    production ``cl100k_base`` packing, so this lane says nothing about production chunk
    counts, corpus fingerprints or overlay topology.

real lane         ``python -m pytest --real-tiktoken -m real_tiktoken``
    Real ``cl100k_base``, loaded from a tiktoken cache that was provisioned beforehand.
    It never downloads, never substitutes the stand-in and never skips: with the vocabulary
    missing it stops at setup. It runs the tests marked ``real_tiktoken``, the production
    tokenization invariants (``tests/unit/test_real_tokenizer_golden.py`` and friends).
"""

from __future__ import annotations

REAL_LANE_COMMAND = "python -m pytest --real-tiktoken -m real_tiktoken"
CL100K = "cl100k_base"

DEFAULT_LANE_NOTE = (
    "tokenizer: offline substitute active (deterministic word-level stand-in, not "
    f"{CL100K}); production tokenization is verified separately with: {REAL_LANE_COMMAND}"
)
REAL_LANE_NOTE = (
    f"tokenizer: real {CL100K} from the local tiktoken cache (downloads refused); "
    "real_tiktoken tests run"
)
SKIP_REASON_DEFAULT_LANE = (
    f"verifies production {CL100K} tokenization, which the offline stand-in does not; "
    f"run the real-tokenizer lane: {REAL_LANE_COMMAND}"
)


class RealTokenizerNotProvisioned(RuntimeError):
    """The real-tokenizer lane was requested but the vocabulary is not in the local cache."""


def provisioning_message() -> str:
    return (
        f"--real-tiktoken needs the real {CL100K} vocabulary in the local tiktoken cache, "
        "and it is not there. Tests never download it.\n"
        "Provision the cache once, outside pytest, from a machine with network access:\n"
        f"    python -c \"import tiktoken; tiktoken.get_encoding('{CL100K}')\"\n"
        "(the cache directory is $TIKTOKEN_CACHE_DIR, else $DATA_GYM_CACHE_DIR, else "
        "<system temp>/data-gym-cache; keep the same variable set when running the lane), "
        f"then run: {REAL_LANE_COMMAND}"
    )


def require_provisioned_cl100k() -> None:
    """Load the real encoding from the local cache, or raise ``RealTokenizerNotProvisioned``.

    tiktoken fetches a missing vocabulary through ``tiktoken.load.read_file``. That function
    is replaced for the duration of the check, so an empty cache fails here without any
    network attempt. Once loaded, the encoding stays in tiktoken's registry and later
    ``get_encoding`` calls are cache hits.
    """
    import tiktoken
    import tiktoken.load
    import pytest

    def refuse_download(blobpath: str) -> bytes:
        raise RealTokenizerNotProvisioned(provisioning_message())

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tiktoken.load, "read_file", refuse_download)
        tiktoken.get_encoding(CL100K)
