#!/usr/bin/env python3
"""
Convert a JSONL file to a human-readable Markdown document.

Usage:
    python scripts/jsonl_to_markdown.py eval_data/medgemma4b_mini.jsonl
    python scripts/jsonl_to_markdown.py @eval_data/medgemma4b_mini.jsonl -o eval_data/mini.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


LONG_TEXT_THRESHOLD = 120
DEFAULT_TEXT_LIMIT = 2000


def _normalize_input_path(path_text: str) -> Path:
    """Accept paths like '@eval_data/foo.jsonl' and return a normal Path."""
    return Path(path_text[1:]) if path_text.startswith("@") else Path(path_text)


def _default_output_path(input_path: Path) -> Path:
    """Create output markdown path next to input JSONL."""
    return input_path.with_suffix(".md")


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _format_scalar(value: Any) -> str:
    if value is None:
        return "`null`"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    if isinstance(value, (int, float)):
        return f"`{value}`"
    text = str(value)
    if "\n" in text or len(text) > LONG_TEXT_THRESHOLD:
        return ""
    return text


def _safe_code_block(text: str) -> str:
    return text.replace("```", "'''")


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n...[truncated]...", True


def _render_value(value: Any, indent: int, text_limit: int) -> list[str]:
    prefix = "  " * indent

    if _is_scalar(value):
        scalar = _format_scalar(value)
        if scalar:
            return [f"{prefix}- {scalar}"]
        text = str(value)
        clipped, was_truncated = _truncate_text(text, text_limit)
        lines = [f"{prefix}```text"]
        lines.extend(f"{prefix}{line}" for line in _safe_code_block(clipped).splitlines())
        lines.append(f"{prefix}```")
        if was_truncated:
            lines.append(f"{prefix}_Text truncated to {text_limit} chars._")
        return lines

    if isinstance(value, dict):
        if not value:
            return [f"{prefix}- `{{}}`"]

        lines: list[str] = []
        for key, inner in value.items():
            if _is_scalar(inner):
                scalar = _format_scalar(inner)
                if scalar:
                    lines.append(f"{prefix}- **{key}**: {scalar}")
                    continue
            lines.append(f"{prefix}- **{key}**:")
            lines.extend(_render_value(inner, indent + 1, text_limit))
        return lines

    if isinstance(value, list):
        if not value:
            return [f"{prefix}- `[]`"]
        lines: list[str] = []
        for item in value:
            lines.extend(_render_value(item, indent, text_limit))
        return lines

    return [f"{prefix}- `{repr(value)}`"]


def _record_title(record: Any, index: int) -> str:
    if isinstance(record, dict):
        for key in ("case_id", "id", "name", "timestamp"):
            val = record.get(key)
            if isinstance(val, (str, int, float)) and str(val).strip():
                return f"{index}. {val}"
    return f"{index}. record_{index:04d}"


def _build_field_stats(records: list[Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for rec in records:
        if isinstance(rec, dict):
            counter.update(rec.keys())
    return counter


def convert_jsonl_to_markdown(input_path: Path, output_path: Path, text_limit: int) -> tuple[int, int]:
    """Convert JSONL records to Markdown and return (record_count, error_count)."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records: list[Any] = []
    errors: list[str] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: {exc.msg}")

    stats = _build_field_stats(records)
    lines: list[str] = []
    lines.append(f"# JSONL Readable Export: `{input_path.name}`")
    lines.append("")
    lines.append(f"- Source: `{input_path}`")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Parsed records: `{len(records)}`")
    lines.append(f"- Parse errors: `{len(errors)}`")
    lines.append("")

    if stats:
        lines.append("## Field Coverage")
        lines.append("")
        lines.append("| Field | Count |")
        lines.append("|---|---:|")
        for field, count in stats.most_common():
            lines.append(f"| `{field}` | {count} |")
        lines.append("")

    if errors:
        lines.append("## Parse Errors")
        lines.append("")
        for err in errors:
            lines.append(f"- {err}")
        lines.append("")

    lines.append("## Records")
    lines.append("")

    for i, record in enumerate(records, start=1):
        lines.append(f"### {_record_title(record, i)}")
        lines.append("")
        if isinstance(record, dict):
            for key, value in record.items():
                if _is_scalar(value):
                    scalar = _format_scalar(value)
                    if scalar:
                        lines.append(f"- **{key}**: {scalar}")
                        continue
                lines.append(f"- **{key}**:")
                lines.extend(_render_value(value, 1, text_limit))
        else:
            lines.extend(_render_value(record, 0, text_limit))
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return len(records), len(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSONL to human-readable Markdown")
    parser.add_argument("input", help="Input JSONL path (supports @prefix, e.g. @eval_data/file.jsonl)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output Markdown path (default: same path with .md suffix)",
    )
    parser.add_argument(
        "--text-limit",
        type=int,
        default=DEFAULT_TEXT_LIMIT,
        help=f"Max chars for long text fields (default: {DEFAULT_TEXT_LIMIT})",
    )
    args = parser.parse_args()

    input_path = _normalize_input_path(args.input)
    output_path = Path(args.output) if args.output else _default_output_path(input_path)

    record_count, error_count = convert_jsonl_to_markdown(
        input_path=input_path,
        output_path=output_path,
        text_limit=args.text_limit,
    )
    print(f"Converted {record_count} records to {output_path}")
    if error_count:
        print(f"Warning: {error_count} lines failed to parse")


if __name__ == "__main__":
    main()
