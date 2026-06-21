"""Central configuration for ingestion and chunking."""

from dataclasses import dataclass
from pathlib import Path

# Default paths (relative to project root)
DEFAULT_INPUT_DIR = Path("data/02_clean_markdown")
DEFAULT_OUTPUT_PATH = Path("data/03_chunks/chunks.jsonl")
DEFAULT_STATS_PATH = Path("data/03_chunks/chunk_stats.json")
DEFAULT_ENCODING = "cl100k_base"

# Token limits — policies, procedures (docs/03_chunking_strategy.md)
POLICY_TARGET_MIN = 450
POLICY_TARGET_MAX = 700
POLICY_SOFT_MIN = 300
POLICY_SOFT_MAX = 800
POLICY_HARD_MAX = 900

# Reference document (01_service_overview)
REFERENCE_TARGET_MIN = 300
REFERENCE_TARGET_MAX = 500
REFERENCE_SOFT_MIN = 300
REFERENCE_SOFT_MAX = 800

# Procedure (07)
PROCEDURE_TARGET_MIN = 400
PROCEDURE_TARGET_MAX = 650
PROCEDURE_SOFT_MIN = 300
PROCEDURE_SOFT_MAX = 800

# Escalation (08)
ESCALATION_TARGET_MIN = 300
ESCALATION_TARGET_MAX = 600
ESCALATION_SOFT_MIN = 300
ESCALATION_SOFT_MAX = 800

# Templates (09)
TEMPLATE_SOFT_MIN = 300
TEMPLATE_SOFT_MAX = 700

# FAQ (10)
FAQ_SOFT_MIN = 150
FAQ_SOFT_MAX = 400

# Overlap
OVERLAP_MIN = 80
OVERLAP_MAX = 120
OVERLAP_MAX_RATIO = 0.20

# Expected FAQ count
EXPECTED_FAQ_COUNT = 45


@dataclass(frozen=True)
class ChunkingLimits:
    """Per-strategy token limits."""

    soft_min: int
    soft_max: int
    hard_max: int = POLICY_HARD_MAX
    target_min: int | None = None
    target_max: int | None = None
    overlap_min: int = 0
    overlap_max: int = 0
    allow_overlap: bool = False


LIMITS_BY_STRATEGY: dict[str, ChunkingLimits] = {
    "reference": ChunkingLimits(
        soft_min=REFERENCE_SOFT_MIN,
        soft_max=REFERENCE_SOFT_MAX,
        target_min=REFERENCE_TARGET_MIN,
        target_max=REFERENCE_TARGET_MAX,
        allow_overlap=False,
    ),
    "policy": ChunkingLimits(
        soft_min=POLICY_SOFT_MIN,
        soft_max=POLICY_SOFT_MAX,
        target_min=POLICY_TARGET_MIN,
        target_max=POLICY_TARGET_MAX,
        overlap_min=OVERLAP_MIN,
        overlap_max=OVERLAP_MAX,
        allow_overlap=True,
    ),
    "procedure": ChunkingLimits(
        soft_min=PROCEDURE_SOFT_MIN,
        soft_max=PROCEDURE_SOFT_MAX,
        target_min=PROCEDURE_TARGET_MIN,
        target_max=PROCEDURE_TARGET_MAX,
        overlap_min=80,
        overlap_max=100,
        allow_overlap=True,
    ),
    "escalation": ChunkingLimits(
        soft_min=ESCALATION_SOFT_MIN,
        soft_max=ESCALATION_SOFT_MAX,
        target_min=ESCALATION_TARGET_MIN,
        target_max=ESCALATION_TARGET_MAX,
        allow_overlap=False,
    ),
    "templates": ChunkingLimits(
        soft_min=TEMPLATE_SOFT_MIN,
        soft_max=TEMPLATE_SOFT_MAX,
        target_min=TEMPLATE_SOFT_MIN,
        target_max=TEMPLATE_SOFT_MAX,
        allow_overlap=False,
    ),
    "faq": ChunkingLimits(
        soft_min=FAQ_SOFT_MIN,
        soft_max=FAQ_SOFT_MAX,
        target_min=FAQ_SOFT_MIN,
        target_max=FAQ_SOFT_MAX,
        allow_overlap=False,
    ),
}

DOCUMENT_STRATEGY_MAP: dict[str, str] = {
    "01_service_overview": "reference",
    "02_delivery_rules": "policy",
    "03_order_changes_and_cancellations": "policy",
    "04_refund_policy": "policy",
    "05_compensation_policy": "policy",
    "06_food_quality_and_packaging": "policy",
    "07_complaint_handling_procedure": "procedure",
    "08_escalation_and_risk_rules": "escalation",
    "09_response_style_and_templates": "templates",
    "10_customer_faq": "faq",
}

DOCUMENT_TYPE_STRATEGY: dict[str, str] = {
    "reference": "reference",
    "policy": "policy",
    "procedure": "procedure",
    "faq": "faq",
    "templates": "templates",
    "guideline": "templates",
}

# --- Retrieval / vector index defaults ---

DEFAULT_INDEX_DIR = Path("data/04_index")
DEFAULT_COLLECTION_NAME = "customer_claims"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 4
DEFAULT_FETCH_K = 12
DEFAULT_SIMILARITY_THRESHOLD = 0.0
DEFAULT_EMBEDDING_BATCH_SIZE = 64
MIN_SIMILARITY_THRESHOLD = 0.0
MAX_SIMILARITY_THRESHOLD = 1.0

INDEX_FORMAT_VERSION = "1.0.0"
METADATA_SCHEMA_VERSION = "1.0.0"
MANIFEST_FILENAME = "manifest.json"
