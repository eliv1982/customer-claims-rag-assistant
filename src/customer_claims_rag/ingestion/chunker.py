"""Hybrid chunking strategies per document type."""

from __future__ import annotations

import re
from dataclasses import dataclass

from customer_claims_rag.config import (
    DOCUMENT_STRATEGY_MAP,
    DOCUMENT_TYPE_STRATEGY,
    LIMITS_BY_STRATEGY,
    OVERLAP_MAX_RATIO,
    ChunkingLimits,
)
from customer_claims_rag.exceptions import ChunkingError
from customer_claims_rag.ingestion.markdown_parser import parse_heading_blocks, parse_h3_subblocks
from customer_claims_rag.models import ChunkMetadata, ChunkRecord, DocumentRecord, HeadingBlock
from customer_claims_rag.token_counter import TokenCounter

_FAQ_HEADING_RE = re.compile(r"^FAQ-(\d+)\.", re.IGNORECASE)
_TEMPLATE_HEADING_RE = re.compile(r"^(\d+)\.\s+")
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")
_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯA-Z«\"])")


def _is_table_separator_row(row: str) -> bool:
    return bool(_TABLE_SEPARATOR_RE.match(row.strip()))


@dataclass
class _PartialChunk:
    heading: str
    heading_path: list[str]
    body_parts: list[str]
    overlap_tokens: int = 0
    semantic_key: str | None = None


class HybridChunker:
    """Apply document-specific chunking strategies."""

    def __init__(
        self,
        token_counter: TokenCounter,
        *,
        fail_on_soft_limit: bool = False,
    ) -> None:
        self.token_counter = token_counter
        self.fail_on_soft_limit = fail_on_soft_limit

    def chunk_document(self, document: DocumentRecord) -> list[ChunkRecord]:
        strategy = self._resolve_strategy(document)
        if strategy == "faq":
            return self._chunk_faq(document)
        if strategy == "templates":
            return self._chunk_templates(document)
        return self._chunk_semantic(document, strategy=strategy)

    def _resolve_strategy(self, document: DocumentRecord) -> str:
        doc_id = document.metadata.document_id
        if doc_id in DOCUMENT_STRATEGY_MAP:
            return DOCUMENT_STRATEGY_MAP[doc_id]
        doc_type = document.metadata.document_type
        return DOCUMENT_TYPE_STRATEGY.get(doc_type, "policy")

    def _build_prefix(self, document: DocumentRecord, heading: str) -> str:
        section = heading or "Общий раздел"
        return (
            f"Документ: {document.metadata.document_id} | Раздел: {section}\n"
            f"---\n"
        )

    def _finalize_chunks(
        self,
        document: DocumentRecord,
        partials: list[_PartialChunk],
        *,
        strategy: str,
        start_index: int = 1,
    ) -> list[ChunkRecord]:
        limits = LIMITS_BY_STRATEGY[strategy]
        records: list[ChunkRecord] = []
        seq = start_index
        for partial in partials:
            body = "\n\n".join(part for part in partial.body_parts if part.strip()).strip()
            if not body or self._is_heading_only(body):
                continue
            prefix = self._build_prefix(document, partial.heading)
            content = prefix + body
            token_count = self.token_counter.count(content)
            self._enforce_limits(
                document,
                content,
                token_count,
                limits,
                strategy=strategy,
                heading=partial.heading,
                atomic=partial.semantic_key is not None
                and (
                    partial.semantic_key.startswith("faq-")
                    or partial.semantic_key.startswith("template-")
                    or partial.semantic_key.startswith("forbidden-")
                    or partial.semantic_key == "forbidden-formulations"
                ),
            )
            if partial.semantic_key:
                chunk_id = f"{document.metadata.document_id}::{partial.semantic_key}"
            else:
                chunk_id = f"{document.metadata.document_id}::chunk-{seq:03d}"
            section = partial.heading or (partial.heading_path[-1] if partial.heading_path else "")
            subsection = (
                partial.heading_path[-1]
                if len(partial.heading_path) > 1 and partial.heading_path[-1] != section
                else None
            )
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=document.metadata.document_id,
                    source_path=document.source_path,
                    title=document.metadata.title,
                    category=document.metadata.category,
                    document_type=document.metadata.document_type,
                    source_type=document.metadata.source_type,
                    document_priority=document.metadata.priority,
                    chunk_index=seq,
                    heading=partial.heading,
                    heading_path=partial.heading_path,
                    content=content,
                    token_count=token_count,
                    word_count=len(content.split()),
                    char_count=len(content),
                    strategy=strategy,
                    overlap_tokens=partial.overlap_tokens,
                    metadata=ChunkMetadata.from_document(
                        document.metadata,
                        source_path=document.source_path,
                        section=section or None,
                        subsection=subsection,
                        topic=partial.semantic_key,
                    ),
                )
            )
            seq += 1
        return records

    def _enforce_limits(
        self,
        document: DocumentRecord,
        content: str,
        token_count: int,
        limits: ChunkingLimits,
        *,
        strategy: str,
        heading: str,
        atomic: bool = False,
    ) -> None:
        if token_count >= limits.hard_max:
            if atomic:
                raise ChunkingError(
                    f"atomic chunk exceeds hard max ({limits.hard_max} tokens, got {token_count}) "
                    f"for strategy {strategy!r}, heading {heading!r}",
                    file_path=document.source_path,
                )
            raise ChunkingError(
                f"chunk exceeds hard max ({limits.hard_max} tokens, got {token_count}) "
                f"for strategy {strategy!r}, heading {heading!r}",
                file_path=document.source_path,
            )
        if self.fail_on_soft_limit and strategy not in {"faq", "templates"}:
            if token_count < limits.soft_min or token_count > limits.soft_max:
                raise ChunkingError(
                    f"chunk outside soft limits ({limits.soft_min}-{limits.soft_max}, "
                    f"got {token_count}) for heading {heading!r}",
                    file_path=document.source_path,
                )

    def _is_heading_only(self, text: str) -> bool:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return bool(lines) and all(line.startswith("#") for line in lines)

    def _chunk_faq(self, document: DocumentRecord) -> list[ChunkRecord]:
        blocks = parse_heading_blocks(document)
        intro_parts: list[str] = []
        partials: list[_PartialChunk] = []

        for block in blocks:
            match = _FAQ_HEADING_RE.match(block.heading)
            if match:
                faq_num = int(match.group(1))
                body = block.content
                if faq_num == 1 and intro_parts:
                    intro_text = "\n\n".join(intro_parts).strip()
                    if intro_text:
                        body = f"{intro_text}\n\n{body}"
                partials.append(
                    _PartialChunk(
                        heading=block.heading,
                        heading_path=block.heading_path,
                        body_parts=[body],
                        semantic_key=f"faq-{faq_num:02d}",
                    )
                )
                continue

            content = block.content.strip()
            if content and not self._is_heading_only(content):
                intro_parts.append(content)

        return self._finalize_chunks(document, partials, strategy="faq")

    def _chunk_templates(self, document: DocumentRecord) -> list[ChunkRecord]:
        blocks = parse_heading_blocks(document)
        partials: list[_PartialChunk] = []

        for block in blocks:
            heading_lower = block.heading.lower()
            if "запретные формулировки" in heading_lower:
                self._validate_negative_example_block(block, document)
                partials.extend(self._chunk_forbidden_formulations(block, document))
                continue

            if block.heading == "Шаблоны проектов ответов":
                subblocks = parse_h3_subblocks(block)
                pending_intro = ""
                for sb in subblocks:
                    match = _TEMPLATE_HEADING_RE.match(sb.heading)
                    if match:
                        content = sb.content
                        if pending_intro:
                            content = f"{pending_intro}\n\n{content}"
                            pending_intro = ""
                        partials.append(
                            _PartialChunk(
                                heading=sb.heading,
                                heading_path=sb.heading_path,
                                body_parts=[content],
                                semantic_key=f"template-{int(match.group(1)):02d}",
                            )
                        )
                    elif sb.content.strip():
                        pending_intro = (
                            f"{pending_intro}\n\n{sb.content}".strip()
                            if pending_intro
                            else sb.content
                        )
                continue

            partials.extend(
                self._split_semantic_block(
                    block,
                    strategy="templates",
                    limits=LIMITS_BY_STRATEGY["templates"],
                )
            )

        records = self._finalize_chunks(document, partials, strategy="templates")
        self._validate_forbidden_rows(records, document)
        self._validate_template_atomicity(records, document)
        return records

    def _validate_forbidden_rows(
        self, records: list[ChunkRecord], document: DocumentRecord
    ) -> None:
        for record in records:
            if "forbidden-" not in record.chunk_id:
                continue
            content_lower = record.content.lower()
            has_negative = "**недопустимо:**" in content_lower or "недопустимо:" in content_lower
            has_reason = self._forbidden_data_row_has_reason(record.content)
            has_safe = "используйте вместо" in content_lower
            has_trailing = "не являются" in content_lower and "шаблон" in content_lower
            if has_negative and not has_reason:
                raise ChunkingError(
                    f"chunk {record.chunk_id!r} missing reason for forbidden formulation",
                    file_path=document.source_path,
                )
            if has_negative and not has_safe:
                raise ChunkingError(
                    f"chunk {record.chunk_id!r} missing safe alternative",
                    file_path=document.source_path,
                )
            if has_negative and not has_trailing:
                raise ChunkingError(
                    f"chunk {record.chunk_id!r} missing trailing safety paragraph",
                    file_path=document.source_path,
                )

    def _forbidden_data_row_has_reason(self, content: str) -> bool:
        for line in content.split("\n"):
            line_stripped = line.strip()
            if not line_stripped.startswith("|"):
                continue
            lower = line_stripped.lower()
            if "недопустимо" not in lower:
                continue
            cells = [cell.strip() for cell in line_stripped.split("|")[1:-1]]
            if len(cells) >= 2:
                return bool(cells[1].strip("* ").strip())
        return False

    def _chunk_forbidden_formulations(
        self, block: HeadingBlock, document: DocumentRecord
    ) -> list[_PartialChunk]:
        """Split forbidden-formulations table by rows; each row stays atomic."""
        lines = block.content.split("\n")
        intro_lines: list[str] = []
        table_lines: list[str] = []
        trailing_lines: list[str] = []
        in_table = False

        for line in lines:
            if line.strip().startswith("|"):
                in_table = True
                table_lines.append(line)
            elif not in_table:
                intro_lines.append(line)
            else:
                trailing_lines.append(line)

        intro = "\n".join(intro_lines).strip()
        trailing = "\n".join(trailing_lines).strip()

        if len(table_lines) < 2:
            raise ChunkingError(
                "forbidden formulations table must contain header and separator rows",
                file_path=document.source_path,
            )

        header = table_lines[0]
        separator = table_lines[1]
        data_rows = [
            row
            for row in table_lines[2:]
            if row.strip() and not _is_table_separator_row(row)
        ]

        if not data_rows:
            raise ChunkingError(
                "forbidden formulations table must contain at least one data row",
                file_path=document.source_path,
            )

        partials: list[_PartialChunk] = []
        for index, row in enumerate(data_rows, start=1):
            table_chunk = f"{header}\n{separator}\n{row}"
            body_parts: list[str] = []
            if intro:
                body_parts.append(intro)
            body_parts.append(table_chunk)
            if trailing:
                body_parts.append(trailing)
            partials.append(
                _PartialChunk(
                    heading=block.heading,
                    heading_path=block.heading_path,
                    body_parts=body_parts,
                    semantic_key=f"forbidden-{index:02d}",
                )
            )
        return partials

    def _validate_negative_example_block(
        self, block: HeadingBlock, document: DocumentRecord
    ) -> None:
        content = block.content.lower()
        required_markers = ["недопустимо", "используйте вместо", "почему нельзя"]
        if not all(marker in content for marker in required_markers):
            raise ChunkingError(
                "forbidden formulations block must contain negative examples with safe alternatives",
                file_path=document.source_path,
            )

    def _validate_template_atomicity(
        self, records: list[ChunkRecord], document: DocumentRecord
    ) -> None:
        for record in records:
            is_template = "::template-" in record.chunk_id
            is_forbidden = "forbidden-" in record.chunk_id or "forbidden-formulations" in record.chunk_id
            if not is_template and not is_forbidden:
                continue
            content_lower = record.content.lower()
            has_negative = "**недопустимо:**" in content_lower or "недопустимо:" in content_lower
            has_safe = "используйте вместо" in content_lower
            if has_negative and not has_safe:
                raise ChunkingError(
                    f"chunk {record.chunk_id!r} contains negative example without safe alternative",
                    file_path=document.source_path,
                )

    def _chunk_semantic(self, document: DocumentRecord, *, strategy: str) -> list[ChunkRecord]:
        blocks = parse_heading_blocks(document)
        limits = LIMITS_BY_STRATEGY[strategy]
        partials: list[_PartialChunk] = []

        for block in blocks:
            subblocks = parse_h3_subblocks(block)
            for subblock in subblocks:
                split_parts = self._split_semantic_block(subblock, strategy=strategy, limits=limits)
                partials.extend(split_parts)

        merged = self._merge_small_blocks(partials, strategy=strategy, limits=limits)
        return self._finalize_chunks(document, merged, strategy=strategy)

    def _merge_small_blocks(
        self,
        partials: list[_PartialChunk],
        *,
        strategy: str,
        limits: ChunkingLimits,
    ) -> list[_PartialChunk]:
        if not partials or strategy in {"faq", "templates"}:
            return partials

        merged: list[_PartialChunk] = []
        index = 0
        while index < len(partials):
            current = partials[index]
            body = "\n\n".join(current.body_parts)
            tokens = self.token_counter.count(body)
            h2_key = current.heading_path[-2] if len(current.heading_path) >= 2 else current.heading

            if tokens < limits.soft_min and index + 1 < len(partials):
                nxt = partials[index + 1]
                nxt_h2 = nxt.heading_path[-2] if len(nxt.heading_path) >= 2 else nxt.heading
                combined_body = "\n\n".join(current.body_parts + nxt.body_parts)
                combined_tokens = self.token_counter.count(combined_body)
                if h2_key == nxt_h2 and combined_tokens <= limits.soft_max:
                    merged.append(
                        _PartialChunk(
                            heading=current.heading,
                            heading_path=current.heading_path,
                            body_parts=current.body_parts + nxt.body_parts,
                            overlap_tokens=0,
                        )
                    )
                    index += 2
                    continue

            merged.append(current)
            index += 1
        return merged

    def _strip_leading_headings(self, text: str) -> str:
        lines = text.split("\n")
        while lines and re.match(r"^#{1,3}\s+", lines[0].strip()):
            lines.pop(0)
        return "\n".join(lines).strip()

    def _split_semantic_block(
        self,
        block: HeadingBlock,
        *,
        strategy: str,
        limits: ChunkingLimits | None = None,
    ) -> list[_PartialChunk]:
        limits = limits or LIMITS_BY_STRATEGY[strategy]
        body = self._strip_leading_headings(block.content.strip())
        if not body:
            return []

        token_count = self.token_counter.count(body)
        if token_count <= limits.soft_max:
            return [
                _PartialChunk(
                    heading=block.heading,
                    heading_path=block.heading_path,
                    body_parts=[body],
                )
            ]

        paragraphs = self._split_paragraphs(body)
        if len(paragraphs) > 1:
            grouped = self._group_paragraphs(
                paragraphs,
                limits=limits,
                heading=block.heading,
                heading_path=block.heading_path,
                strategy=strategy,
            )
            if grouped:
                return grouped

        sentences = self._split_sentences(body)
        if len(sentences) > 1:
            grouped = self._group_sentences(
                sentences,
                limits=limits,
                heading=block.heading,
                heading_path=block.heading_path,
                strategy=strategy,
            )
            if grouped:
                return grouped

        if limits.allow_overlap:
            return self._fixed_token_split(
                body,
                limits=limits,
                heading=block.heading,
                heading_path=block.heading_path,
            )

        return [
            _PartialChunk(
                heading=block.heading,
                heading_path=block.heading_path,
                body_parts=[body],
            )
        ]

    def _split_paragraphs(self, text: str) -> list[str]:
        parts = re.split(r"\n\s*\n", text)
        result: list[str] = []
        buffer: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if _TABLE_ROW_RE.match(part.split("\n")[0]):
                if buffer:
                    result.append("\n\n".join(buffer))
                    buffer = []
                table_lines = [part]
                result.append("\n".join(table_lines))
                continue
            buffer.append(part)
        if buffer:
            result.append("\n\n".join(buffer))
        return result

    def _group_paragraphs(
        self,
        paragraphs: list[str],
        *,
        limits: ChunkingLimits,
        heading: str,
        heading_path: list[str],
        strategy: str,
    ) -> list[_PartialChunk]:
        groups: list[_PartialChunk] = []
        current: list[str] = []

        def flush(overlap: int = 0) -> None:
            if current:
                groups.append(
                    _PartialChunk(
                        heading=heading,
                        heading_path=heading_path,
                        body_parts=["\n\n".join(current)],
                        overlap_tokens=overlap,
                    )
                )

        for para in paragraphs:
            candidate = current + [para]
            tokens = self.token_counter.count("\n\n".join(candidate))
            if tokens <= limits.soft_max or not current:
                current.append(para)
            else:
                flush()
                current = [para]

        if current:
            tokens = self.token_counter.count("\n\n".join(current))
            if tokens > limits.soft_max and limits.allow_overlap:
                return self._fixed_token_split(
                    "\n\n".join(paragraphs),
                    limits=limits,
                    heading=heading,
                    heading_path=heading_path,
                )
            flush()
        return groups

    def _split_sentences(self, text: str) -> list[str]:
        lines = text.split("\n")
        sentences: list[str] = []
        for line in lines:
            if _TABLE_ROW_RE.match(line) or line.strip().startswith("#"):
                sentences.append(line)
                continue
            if line.strip().startswith(("-", "*", "1.")):
                sentences.append(line)
                continue
            parts = _SENTENCE_SPLIT_RE.split(line.strip())
            sentences.extend(part for part in parts if part.strip())
        return sentences

    def _group_sentences(
        self,
        sentences: list[str],
        *,
        limits: ChunkingLimits,
        heading: str,
        heading_path: list[str],
        strategy: str,
    ) -> list[_PartialChunk]:
        groups: list[_PartialChunk] = []
        current: list[str] = []

        def flush(overlap: int = 0) -> None:
            if current:
                groups.append(
                    _PartialChunk(
                        heading=heading,
                        heading_path=heading_path,
                        body_parts=["\n".join(current)],
                        overlap_tokens=overlap,
                    )
                )

        for sentence in sentences:
            candidate = current + [sentence]
            tokens = self.token_counter.count("\n".join(candidate))
            if tokens <= limits.soft_max or not current:
                current.append(sentence)
            else:
                flush()
                current = [sentence]

        if current:
            tokens = self.token_counter.count("\n".join(current))
            if tokens > limits.soft_max and limits.allow_overlap:
                return self._fixed_token_split(
                    "\n".join(sentences),
                    limits=limits,
                    heading=heading,
                    heading_path=heading_path,
                )
            flush()
        return groups

    def _fixed_token_split(
        self,
        text: str,
        *,
        limits: ChunkingLimits,
        heading: str,
        heading_path: list[str],
    ) -> list[_PartialChunk]:
        overlap = limits.overlap_min
        # Fixed overlap at the documented lower bound of the 80-120 token target range.
        max_ratio_overlap = int(limits.soft_max * OVERLAP_MAX_RATIO)
        overlap = min(overlap, max_ratio_overlap)

        windows = self.token_counter.split_by_token_window(
            text,
            max_tokens=limits.soft_max,
            overlap_tokens=overlap,
        )
        result: list[_PartialChunk] = []
        for idx, window in enumerate(windows):
            overlap_count = 0
            if idx > 0:
                prev_text = windows[idx - 1]
                max_len = min(len(prev_text), len(window))
                shared_len = 0
                for length in range(max_len, 0, -1):
                    if window.startswith(prev_text[-length:]):
                        shared_len = length
                        break
                if shared_len:
                    overlap_count = self.token_counter.count(prev_text[-shared_len:])
                overlap_count = min(overlap_count, overlap)
            result.append(
                _PartialChunk(
                    heading=heading,
                    heading_path=heading_path,
                    body_parts=[window],
                    overlap_tokens=overlap_count,
                )
            )
        return result
