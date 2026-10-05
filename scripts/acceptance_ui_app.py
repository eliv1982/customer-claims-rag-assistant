"""Streamlit entry point for the acceptance screenshots (evidence tooling, not the product entry point).

The real UI (``customer_claims_rag.ui.streamlit_app``) over the real release gate and the real
production index, with the two paid boundaries stubbed per recorded scenario exactly as in
``scripts/capture_final_acceptance.py``. It serves only the recorded scenario messages and needs no
credential. Start it through ``scripts/capture_ui_screenshots.py``; by hand:

    streamlit run scripts/acceptance_ui_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

import capture_final_acceptance as acceptance  # noqa: E402
from customer_claims_rag.ui import streamlit_app  # noqa: E402


@st.cache_resource
def _scenario_pipeline():
    acceptance.prepare_environment()
    return acceptance.Harness().router()


# The UI asks this resource for its pipeline; everything else (release identity, rendering, the
# customer-output projection) is the unmodified application code.
streamlit_app.get_production_pipeline = _scenario_pipeline
streamlit_app.main()
