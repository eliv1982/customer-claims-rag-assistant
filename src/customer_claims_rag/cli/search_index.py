"""Search the persistent vector index."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from customer_claims_rag.config import DEFAULT_INDEX_DIR
from customer_claims_rag.env_bootstrap import load_project_env
from customer_claims_rag.exceptions import RetrievalError
from customer_claims_rag.retrieval.factory import create_embedding_provider, create_vector_store
from customer_claims_rag.retrieval.models import SearchResponse
from customer_claims_rag.retrieval.path_helpers import validate_index_dir
from customer_claims_rag.retrieval.ports import VectorStore
from customer_claims_rag.retrieval.retriever import BaselineRetriever
from customer_claims_rag.retrieval_config import RetrievalSettings


def build_parser() -> argparse.ArgumentParser:
    settings = RetrievalSettings.from_env()
    parser = argparse.ArgumentParser(
        description="Search the FoodFlow knowledge base vector index.",
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Search query text",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=settings.index_dir,
        help=f"Vector index directory (default: {settings.index_dir})",
    )
    parser.add_argument(
        "--collection",
        default=settings.collection_name,
        help=f"Chroma collection name (default: {settings.collection_name})",
    )
    parser.add_argument(
        "--embedding-model",
        default=settings.embedding_model,
        help=f"OpenAI embedding model (default: {settings.embedding_model})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Maximum results to return (default: {settings.top_k})",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=None,
        help=f"Candidate pool size (default: {settings.fetch_k})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            f"Minimum cosine similarity; default {settings.similarity_threshold} "
            f"disables threshold filtering in baseline retrieval"
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print diagnostic traceback on error",
    )
    return parser


def _format_excerpt(content: str, limit: int = 160) -> str:
    compact = " ".join(content.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _print_human(response: SearchResponse) -> None:
    if not response.results:
        print("No results above threshold")
        print(
            f"Fetched {response.candidates_fetched} candidates; "
            f"{response.candidates_above_threshold} met threshold "
            f"{response.similarity_threshold:.2f}"
        )
        return

    for item in response.results:
        print(f"#{item.rank} similarity={item.similarity:.4f}")
        print(f"  chunk_id: {item.chunk_id}")
        print(f"  document_id: {item.document_id}")
        print(f"  heading: {item.heading}")
        print(f"  source_path: {item.source_path}")
        print(f"  excerpt: {_format_excerpt(item.content)}")


def run_search(
    *,
    query: str,
    index_dir: Path,
    collection_name: str,
    embedding_model: str,
    top_k: int | None = None,
    fetch_k: int | None = None,
    threshold: float | None = None,
    json_output: bool = False,
    embedding_provider_factory=create_embedding_provider,
    vector_store_factory=create_vector_store,
    settings: RetrievalSettings | None = None,
    project_root: Path | None = None,
) -> tuple[int, SearchResponse | None]:
    resolved_settings = settings or RetrievalSettings.from_env()
    vector_store: VectorStore | None = None
    try:
        resolved_top_k, resolved_fetch_k, resolved_threshold = (
            resolved_settings.validate_search_params(
                top_k=top_k,
                fetch_k=fetch_k,
                similarity_threshold=threshold,
            )
        )
        resolved_root = (project_root or Path.cwd()).resolve()
        resolved_index_dir = validate_index_dir(
            index_dir,
            project_root=resolved_root,
        )
        embedding_provider = embedding_provider_factory(
            model_name=embedding_model,
            api_key=resolved_settings.openai_api_key,
        )
        vector_store = vector_store_factory(
            index_dir=resolved_index_dir,
            collection_name=collection_name,
        )
        retriever = BaselineRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            index_dir=resolved_index_dir,
            top_k=resolved_top_k,
            fetch_k=resolved_fetch_k,
            similarity_threshold=resolved_threshold,
        )
        response = retriever.search(query)
    except RetrievalError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1, None
    finally:
        if vector_store is not None:
            vector_store.close()

    if json_output:
        print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        _print_human(response)
    return 0, response


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = RetrievalSettings.from_env()
    query = " ".join(args.query)

    try:
        exit_code, _ = run_search(
            query=query,
            index_dir=args.index_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            threshold=args.threshold,
            json_output=args.json_output,
            settings=settings,
            project_root=Path.cwd().resolve(),
        )
        return exit_code
    except Exception:
        print("Error: search failed", file=sys.stderr)
        if args.verbose:
            try:
                raise RetrievalError("search failed") from None
            except RetrievalError:
                traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
