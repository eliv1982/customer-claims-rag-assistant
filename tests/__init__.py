"""Test suite package.

Makes ``tests.*`` imports (``tests.retrieval_helpers`` ...) resolve the same way
for ``pytest`` and ``python -m pytest``: with this file pytest's default
``prepend`` import mode puts the repository root on ``sys.path``.
"""
