"""Orchestrate loading, chunking, stats and JSONL export."""

from __future__ import annotations

import json
import statistics
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from customer_claims_rag.config import LIMITS_BY_STRATEGY
from customer_claims_rag.exceptions import DuplicateChunkIdError, IngestionError
from customer_claims_rag.ingestion.chunker import HybridChunker
from customer_claims_rag.ingestion.markdown_loader import MarkdownLoader
from customer_claims_rag.ingestion.path_helpers import resolve_safe_path, validate_export_paths
from customer_claims_rag.models import ChunkRecord, CorpusStats, DocumentRecord
from customer_claims_rag.token_counter import TiktokenCounter, TokenCounter


class CorpusBuilder:
    """Build chunk corpus from clean Markdown documents."""

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        include_inactive: bool = False,
        fail_on_soft_limit: bool = False,
        permitted_root: Path | None = None,
    ) -> None:
        if permitted_root is None:
            raise IngestionError("permitted_root is required for CorpusBuilder")
        self.permitted_root = permitted_root.resolve()
        self.loader = MarkdownLoader(
            include_inactive=include_inactive,
            permitted_root=self.permitted_root,
        )
        self.token_counter = token_counter or TiktokenCounter()
        self.chunker = HybridChunker(
            self.token_counter,
            fail_on_soft_limit=fail_on_soft_limit,
        )
        self.fail_on_soft_limit = fail_on_soft_limit

    def build_from_directory(self, input_dir: Path) -> tuple[list[DocumentRecord], list[ChunkRecord]]:
        documents = self.loader.load_directory(input_dir)
        chunks: list[ChunkRecord] = []
        for document in documents:
            chunks.extend(self.chunker.chunk_document(document))
        self.ensure_unique_chunk_ids(chunks)
        return documents, chunks

    def build_from_selection(
        self,
        input_dir: Path,
        file_names: Sequence[str],
    ) -> tuple[list[DocumentRecord], list[ChunkRecord]]:
        """Build only the named documents of ``input_dir``, in the order given.

        Unlike ``build_from_directory`` nothing is discovered: a file that is not named is never
        read, and a named file that is missing or filtered out by status is an error. This is the
        entry point for an explicit corpus selection (``ingestion/canonical_corpus.py``).
        """
        safe_dir = resolve_safe_path(input_dir, root=self.permitted_root)
        documents: list[DocumentRecord] = []
        for name in file_names:
            record = self.loader.load_file(safe_dir / name)
            if record is None:
                raise IngestionError(
                    f"selected document {name!r} was filtered out by its status; "
                    "a corpus selection must name only loadable documents",
                    file_path=str(safe_dir / name),
                )
            documents.append(record)
        if not documents:
            raise IngestionError("corpus selection is empty", file_path=str(safe_dir))
        self.loader.validator.check_duplicate_document_ids(
            [(record.metadata.document_id, record.source_path) for record in documents]
        )
        chunks: list[ChunkRecord] = []
        for document in documents:
            chunks.extend(self.chunker.chunk_document(document))
        self.ensure_unique_chunk_ids(chunks)
        return documents, chunks

    def ensure_unique_chunk_ids(self, chunks: list[ChunkRecord]) -> None:
        seen: dict[str, str] = {}
        for chunk in chunks:
            if chunk.chunk_id in seen:
                raise DuplicateChunkIdError(
                    f"duplicate chunk_id {chunk.chunk_id!r} "
                    f"(already used in {seen[chunk.chunk_id]})",
                    file_path=chunk.source_path,
                )
            seen[chunk.chunk_id] = chunk.source_path

    def compute_stats(
        self,
        documents: list[DocumentRecord],
        chunks: list[ChunkRecord],
    ) -> CorpusStats:
        token_counts = [chunk.token_count for chunk in chunks]
        chunks_by_document = Counter(chunk.document_id for chunk in chunks)
        chunks_by_document_type = Counter(chunk.document_type for chunk in chunks)
        strategy_counts = Counter(chunk.strategy for chunk in chunks)

        soft_violations_below = 0
        soft_violations_above = 0
        hard_violations = 0

        for chunk in chunks:
            limits = LIMITS_BY_STRATEGY.get(chunk.strategy, LIMITS_BY_STRATEGY["policy"])
            if chunk.token_count < limits.soft_min:
                soft_violations_below += 1
            if chunk.token_count > limits.soft_max:
                if chunk.strategy not in {"faq", "templates"}:
                    soft_violations_above += 1
            if chunk.token_count >= limits.hard_max:
                hard_violations += 1

        faq_chunk_count = sum(
            1
            for chunk in chunks
            if chunk.document_id == "10_customer_faq"
            and "::faq-" in chunk.chunk_id
        )
        template_chunk_count = sum(
            1
            for chunk in chunks
            if chunk.document_id == "09_response_style_and_templates"
            and chunk.chunk_id.startswith("09_response_style_and_templates::template-")
        )

        return CorpusStats(
            documents_total=len(documents),
            chunks_total=len(chunks),
            chunks_by_document=dict(sorted(chunks_by_document.items())),
            chunks_by_document_type=dict(sorted(chunks_by_document_type.items())),
            strategy_counts=dict(sorted(strategy_counts.items())),
            min_tokens=min(token_counts) if token_counts else 0,
            max_tokens=max(token_counts) if token_counts else 0,
            average_tokens=statistics.mean(token_counts) if token_counts else 0.0,
            median_tokens=statistics.median(token_counts) if token_counts else 0.0,
            chunks_below_soft_min=soft_violations_below,
            chunks_above_soft_max=soft_violations_above,
            chunks_at_or_above_hard_max=hard_violations,
            overlap_chunks=sum(1 for chunk in chunks if chunk.overlap_tokens > 0),
            faq_chunk_count=faq_chunk_count,
            template_chunk_count=template_chunk_count,
        )

    def _write_jsonl_temp(self, chunks: list[ChunkRecord], temp_path: Path) -> None:
        sorted_chunks = sorted(chunks, key=lambda c: (c.document_id, c.chunk_index))
        with temp_path.open("w", encoding="utf-8") as handle:
            for chunk in sorted_chunks:
                line = chunk.model_dump(mode="json")
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _write_stats_temp(self, stats: CorpusStats, temp_path: Path) -> None:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(stats.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def export_jsonl(self, chunks: list[ChunkRecord], output_path: Path) -> None:
        """Write chunks to JSONL atomically."""
        output_resolved = resolve_safe_path(output_path, root=self.permitted_root)
        output_resolved.parent.mkdir(parents=True, exist_ok=True)

        fd, temp_name = tempfile.mkstemp(
            suffix=".jsonl",
            dir=str(output_resolved.parent),
            text=True,
        )
        temp_path = Path(temp_name)
        try:
            import os

            os.close(fd)
            self._write_jsonl_temp(chunks, temp_path)
            temp_path.replace(output_resolved)
        except Exception:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise

    def export_stats(self, stats: CorpusStats, stats_path: Path) -> None:
        """Write stats JSON atomically."""
        stats_resolved = resolve_safe_path(stats_path, root=self.permitted_root)
        stats_resolved.parent.mkdir(parents=True, exist_ok=True)

        fd, temp_name = tempfile.mkstemp(
            suffix=".json",
            dir=str(stats_resolved.parent),
            text=True,
        )
        temp_path = Path(temp_name)
        try:
            import os

            os.close(fd)
            self._write_stats_temp(stats, temp_path)
            temp_path.replace(stats_resolved)
        except Exception:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise

    def build_and_export(
        self,
        input_dir: Path,
        output_path: Path,
        stats_path: Path,
    ) -> tuple[list[DocumentRecord], list[ChunkRecord], CorpusStats]:
        documents, chunks = self.build_from_directory(input_dir)
        return self.export_corpus(
            documents,
            chunks,
            input_dir=input_dir,
            output_path=output_path,
            stats_path=stats_path,
        )

    def export_corpus(
        self,
        documents: list[DocumentRecord],
        chunks: list[ChunkRecord],
        *,
        input_dir: Path,
        output_path: Path,
        stats_path: Path,
    ) -> tuple[list[DocumentRecord], list[ChunkRecord], CorpusStats]:
        """Write already built documents/chunks (JSONL + stats) atomically."""
        stats = self.compute_stats(documents, chunks)
        source_paths = [document.source_path for document in documents]

        output_resolved, stats_resolved = validate_export_paths(
            permitted_root=self.permitted_root,
            input_dir=input_dir,
            output_path=output_path,
            stats_path=stats_path,
            source_paths=source_paths,
        )

        jsonl_temp_path: Path | None = None
        stats_temp_path: Path | None = None
        try:
            jsonl_fd, jsonl_temp_name = tempfile.mkstemp(
                suffix=".jsonl",
                dir=str(output_resolved.parent),
                text=True,
            )
            stats_fd, stats_temp_name = tempfile.mkstemp(
                suffix=".json",
                dir=str(stats_resolved.parent),
                text=True,
            )
            import os

            os.close(jsonl_fd)
            os.close(stats_fd)
            jsonl_temp_path = Path(jsonl_temp_name)
            stats_temp_path = Path(stats_temp_name)

            self._write_jsonl_temp(chunks, jsonl_temp_path)
            self._write_stats_temp(stats, stats_temp_path)

            jsonl_temp_path.replace(output_resolved)
            jsonl_temp_path = None
            stats_temp_path.replace(stats_resolved)
            stats_temp_path = None
        except Exception:
            for temp in (jsonl_temp_path, stats_temp_path):
                if temp is not None and temp.exists():
                    temp.unlink(missing_ok=True)
            raise

        return documents, chunks, stats
