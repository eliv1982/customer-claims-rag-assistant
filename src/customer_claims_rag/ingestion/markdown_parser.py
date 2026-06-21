"""Predictable Markdown heading parser for H1/H2/H3."""

from __future__ import annotations

import re

from customer_claims_rag.exceptions import MarkdownStructureError
from customer_claims_rag.models import DocumentRecord, HeadingBlock

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```+|~~~)")


def _is_heading_line(line: str, *, in_code_fence: bool) -> re.Match[str] | None:
    if in_code_fence:
        return None
    return _HEADING_RE.match(line)


def parse_heading_blocks(document: DocumentRecord) -> list[HeadingBlock]:
    """Parse document body into heading-bound semantic blocks."""
    lines = document.content.split("\n")
    blocks: list[HeadingBlock] = []
    order = 0

    h1_heading: str | None = None
    current_level = 0
    current_heading = ""
    current_path: list[str] = []
    current_lines: list[str] = []
    in_code_fence = False
    fence_marker: str | None = None
    saw_h2_or_h3 = False

    def flush() -> None:
        nonlocal order
        content = "\n".join(current_lines).strip()
        if not content:
            return
        if current_level == 1:
            return
        non_empty_lines = [line.strip() for line in content.split("\n") if line.strip()]
        if non_empty_lines and all(line.startswith("#") for line in non_empty_lines):
            return
        blocks.append(
            HeadingBlock(
                level=current_level if current_level else 2,
                heading=current_heading,
                heading_path=list(current_path),
                content=content,
                order=order,
            )
        )
        order += 1

    for line in lines:
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[:3]
            if not in_code_fence:
                in_code_fence = True
                fence_marker = marker
            elif line.strip().startswith(fence_marker or marker):
                in_code_fence = False
                fence_marker = None
            current_lines.append(line)
            continue

        heading_match = _is_heading_line(line, in_code_fence=in_code_fence)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            if level > 3:
                current_lines.append(line)
                continue

            if level == 1:
                h1_heading = heading_text
                continue

            if level == 2:
                saw_h2_or_h3 = True
                flush()
                current_level = 2
                current_heading = heading_text
                current_path = [h1_heading, heading_text] if h1_heading else [heading_text]
                current_path = [part for part in current_path if part]
                current_lines = [line]
                continue

            if level == 3:
                saw_h2_or_h3 = True
                if current_level == 0:
                    current_level = 2
                    current_heading = heading_text
                    current_path = [h1_heading, heading_text] if h1_heading else [heading_text]
                    current_path = [part for part in current_path if part]
                    current_lines = [line]
                else:
                    current_lines.append(line)
                    if current_path:
                        if len(current_path) >= 2:
                            current_path = [current_path[0], current_path[1], heading_text]
                        else:
                            current_path = current_path + [heading_text]
                continue

        current_lines.append(line)
        if current_level == 0 and current_lines and not current_heading:
            current_level = 2
            current_heading = ""
            current_path = [h1_heading] if h1_heading else []

    flush()

    if not saw_h2_or_h3:
        raise MarkdownStructureError(
            "document has no H2 or H3 headings",
            file_path=document.source_path,
        )

    if not blocks:
        raise MarkdownStructureError(
            "no semantic blocks found in document",
            file_path=document.source_path,
        )

    return blocks


def parse_h3_subblocks(block: HeadingBlock) -> list[HeadingBlock]:
    """Split an H2 block into H3 sub-blocks when H3 headings are present."""
    lines = block.content.split("\n")
    subblocks: list[HeadingBlock] = []
    order = block.order
    current_heading = block.heading
    current_path = list(block.heading_path)
    current_lines: list[str] = []
    in_code_fence = False
    fence_marker: str | None = None
    has_h3 = False

    for line in lines:
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[:3]
            if not in_code_fence:
                in_code_fence = True
                fence_marker = marker
            elif line.strip().startswith(fence_marker or marker):
                in_code_fence = False
                fence_marker = None
            current_lines.append(line)
            continue

        heading_match = _is_heading_line(line, in_code_fence=in_code_fence)
        if heading_match and len(heading_match.group(1)) == 3:
            has_h3 = True
            content = "\n".join(current_lines).strip()
            if content:
                subblocks.append(
                    HeadingBlock(
                        level=2 if current_heading == block.heading else 3,
                        heading=current_heading,
                        heading_path=list(current_path),
                        content=content,
                        order=order,
                    )
                )
                order += 1
            h3_text = heading_match.group(2).strip()
            current_heading = h3_text
            base = block.heading_path[:2] if len(block.heading_path) >= 2 else block.heading_path
            current_path = base + [h3_text]
            current_lines = [line]
            continue
        current_lines.append(line)

    final_content = "\n".join(current_lines).strip()
    if final_content:
        subblocks.append(
            HeadingBlock(
                level=3 if has_h3 else 2,
                heading=current_heading or block.heading,
                heading_path=current_path if current_path else block.heading_path,
                content=final_content,
                order=order,
            )
        )

    return subblocks if has_h3 else [block]
