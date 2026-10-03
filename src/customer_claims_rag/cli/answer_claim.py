"""Answer a single customer claim via the production application pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from customer_claims_rag.application.customer_output import (
    CustomerOutput,
    SourceReference,
    build_customer_output,
)
from customer_claims_rag.application.factory import build_customer_claims_pipeline
from customer_claims_rag.application.models import (
    CustomerClaimsRequest,
    CustomerClaimsResult,
    is_query_too_long_error,
)
from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
from customer_claims_rag.application.settings import ApplicationSettings
from customer_claims_rag.env_bootstrap import load_project_env
from customer_claims_rag.exceptions import GenerationError, RetrievalError
from customer_claims_rag.input_limits import MAX_CUSTOMER_QUERY_CHARS

_INPUT_ERROR_MESSAGE = "Error: message must not be empty or whitespace-only"
_INPUT_TOO_LONG_MESSAGE = f"Error: message must not exceed {MAX_CUSTOMER_QUERY_CHARS} characters"
_RUNTIME_ERROR_MESSAGE = "Error: claim handling failed"
_CONFIGURATION_ERROR_MESSAGE = "Error: application configuration is invalid"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer a single customer claim using the production RAG pipeline.",
    )
    parser.add_argument(
        "--message",
        required=True,
        help="Customer claim message text",
    )
    return parser


def _source_payload(source: SourceReference) -> dict[str, str]:
    return {"key": source.key, "heading": source.heading, "document_id": source.document_id}


def serialize_customer_output(output: CustomerOutput) -> dict[str, Any]:
    """Serialize the customer-safe projection into the stable CLI JSON schema.

    ``answer`` is exactly the text the UI shows. ``answer_provenance`` says why that text exists
    and ``citations`` are only the sources supporting it (empty unless the answer is the accepted
    grounded model draft); fragments that merely were retrieved are listed separately in
    ``retrieved_materials``. ``risk_floor`` is the raw deterministic lower bound and is only an
    affirmative level when ``assessment_status`` is ``rule_match``; ``risk_label`` is the
    customer-facing wording of the same status.
    """
    return {
        "answer": output.customer_text,
        "answer_provenance": output.provenance,
        "response_mode": output.response_mode,
        "generation_outcome": output.generation_outcome,
        "failure_source": output.failure_source,
        "assessment_status": output.assessment_status.value,
        "risk_floor": output.risk_floor.value,
        "risk_label": output.risk_label,
        "risk_note": output.risk_note,
        "handoff_required": output.handoff_required,
        "priority_handoff": output.priority_handoff,
        "handoff_notice": output.handoff_notice,
        "citations": [_source_payload(source) for source in output.answer_sources],
        "retrieved_materials": [_source_payload(source) for source in output.retrieved_materials],
    }


def format_answer_payload(result: CustomerClaimsResult) -> dict[str, Any]:
    """Map a pipeline result to the stable CLI JSON schema via the customer-output policy."""
    return serialize_customer_output(build_customer_output(result))


def serialize_answer_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def run_answer(
    *,
    message: str,
    settings_loader: Callable[[], ApplicationSettings] | None = None,
    pipeline_factory: Callable[
        [ApplicationSettings],
        CustomerClaimsPipeline,
    ] | None = None,
) -> tuple[int, dict[str, Any] | None]:
    resolve_settings_loader = (
        settings_loader if settings_loader is not None else ApplicationSettings.from_env
    )
    resolve_pipeline_factory = (
        pipeline_factory if pipeline_factory is not None else build_customer_claims_pipeline
    )
    try:
        request = CustomerClaimsRequest(customer_query=message)
    except ValidationError as exc:
        if is_query_too_long_error(exc):
            print(_INPUT_TOO_LONG_MESSAGE, file=sys.stderr)
        else:
            print(_INPUT_ERROR_MESSAGE, file=sys.stderr)
        return 2, None

    try:
        settings = resolve_settings_loader()
        pipeline = resolve_pipeline_factory(settings)
        result = pipeline.handle(request)
    except RetrievalError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1, None
    except GenerationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1, None
    except ValidationError:
        print(_CONFIGURATION_ERROR_MESSAGE, file=sys.stderr)
        return 1, None
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1, None
    except Exception:
        print(_RUNTIME_ERROR_MESSAGE, file=sys.stderr)
        return 1, None

    try:
        payload = format_answer_payload(result)
    except Exception:
        print(_RUNTIME_ERROR_MESSAGE, file=sys.stderr)
        return 1, None
    print(serialize_answer_payload(payload))
    return 0, payload


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code, _ = run_answer(message=args.message)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
