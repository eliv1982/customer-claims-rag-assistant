"""Product-wide input limits shared by every entry point (UI, CLI, library callers)."""

from __future__ import annotations

# Hard upper bound for a single customer message, counted in Unicode code points
# after surrounding whitespace is stripped. Enforced by the domain and application
# request models so that no caller can reach the safety rules or retrieval with an
# oversized message.
MAX_CUSTOMER_QUERY_CHARS = 4000
