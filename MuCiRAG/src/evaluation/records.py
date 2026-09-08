"""Read evaluation datasets and checkpoint formats without plotting dependencies.

Row loading preserves input order. Callers own their duplicate-ID and scoring
policies: summary/comparison reject duplicates, while Venn uses the final row.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

OPTION_PATTERN = re.compile(r"\boption\s*(\d+)\b", re.IGNORECASE)


def option_number(value: object) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if not isinstance(value, str):
        return None
    match = OPTION_PATTERN.search(value)
    return int(match.group(1)) if match else None


def _jsonl_records(text: str, path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(payload)
    return records


def load_json_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL checkpoints plus small legacy JSON/list variants."""
    if not path.is_file():
        raise FileNotFoundError(f"Result file does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return _jsonl_records(text, path)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Older Telco-RAG_api checkpoints sometimes use a .json suffix for
        # JSONL content, so inspect the content instead of trusting the name.
        return _jsonl_records(text, path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object or list at {path}")
    if isinstance(payload.get("question_id"), str):
        return [payload]
    for key in ("records", "results", "rows"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    if all(isinstance(value, dict) for value in payload.values()):
        return [dict(value, question_id=key) for key, value in payload.items() if isinstance(key, str)]
    raise ValueError(f"Cannot find benchmark rows in {path}; use JSONL or a records/results list.")


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"TeleQnA dataset must be an object keyed by question_id: {path}")
    return {key: value for key, value in payload.items() if isinstance(key, str) and isinstance(value, dict)}


def load_final_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")

    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                question_id = record["question_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Malformed record at {path}:{line_number}: {exc}") from exc
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"Invalid question_id at {path}:{line_number}: {question_id!r}")
            records[question_id] = record
    if not records:
        raise ValueError(f"No records found in {path}")
    return records
