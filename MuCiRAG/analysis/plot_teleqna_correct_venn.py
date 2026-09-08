#!/usr/bin/env python3
"""Plot exact overlaps among the correct answers of three TeleQnA runs.

The comparison is by ``question_id``, never JSONL row position.  A resumed
checkpoint may contain an ID more than once; its last occurrence is the final
result.  All three inputs must cover the same expected IDs (1,840 by default),
so an in-progress run cannot silently produce a misleading Venn diagram.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from MuCiRAG.src.evaluation.records import load_final_records as final_records

# This also works on locked-down hosts where ~/.config/matplotlib is read-only.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "teleqna-venn-matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


DEFAULT_EXPECTED_COUNT = 1840
COLORS = ("#4C78A8", "#F58518", "#54A24B")


@dataclass(frozen=True)
class Run:
    label: str
    path: Path
    correct_ids: frozenset[str]
    all_ids: frozenset[str]


def parse_run(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use LABEL=PATH for --run.") from exc
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Both LABEL and PATH must be non-empty.")
    return label.strip(), Path(raw_path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        metavar="LABEL=PATH",
        required=True,
        type=parse_run,
        help="Exactly three labelled benchmark JSONL files. Repeat three times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination PNG (a matching .json overlap summary is also written).",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=DEFAULT_EXPECTED_COUNT,
        help=f"Required unique question IDs per input (default: {DEFAULT_EXPECTED_COUNT}).",
    )
    return parser.parse_args()


def is_correct(record: dict[str, Any]) -> bool:
    """Support the MuCiRAG field and the legacy harness's ``correct`` field."""
    return record.get("is_correct") is True or record.get("correct") is True


def load_run(label: str, path: Path) -> Run:
    records = final_records(path)
    return Run(
        label=label,
        path=path,
        correct_ids=frozenset(question_id for question_id, record in records.items() if is_correct(record)),
        all_ids=frozenset(records),
    )


def validate_runs(runs: Iterable[Run], expected_count: int) -> list[Run]:
    result = list(runs)
    if len(result) != 3:
        raise ValueError("Provide exactly three --run LABEL=PATH arguments.")
    labels = [run.label for run in result]
    if len(set(labels)) != 3:
        raise ValueError("Run labels must be unique.")
    if expected_count <= 0:
        raise ValueError("--expected-count must be positive.")

    reference_ids = result[0].all_ids
    for run in result:
        if len(run.all_ids) != expected_count:
            raise ValueError(
                f"{run.label!r} has {len(run.all_ids)} final unique question IDs; "
                f"expected {expected_count}. It may be incomplete."
            )
        if run.all_ids != reference_ids:
            missing = sorted(reference_ids - run.all_ids)
            extra = sorted(run.all_ids - reference_ids)
            raise ValueError(
                f"Question-ID coverage differs for {run.label!r}; "
                f"missing={missing[:5]}, extra={extra[:5]}."
            )
    return result


def region_counts(runs: list[Run]) -> dict[str, int]:
    a, b, c = (run.correct_ids for run in runs)
    return {
        "100": len(a - b - c),
        "010": len(b - a - c),
        "001": len(c - a - b),
        "110": len((a & b) - c),
        "101": len((a & c) - b),
        "011": len((b & c) - a),
        "111": len(a & b & c),
    }


def plot(runs: list[Run], counts: dict[str, int], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 8), layout="constrained")
    centers = ((-1.0, 0.35), (1.0, 0.35), (0.0, -0.95))
    for center, color in zip(centers, COLORS, strict=True):
        axis.add_patch(Circle(center, radius=1.55, facecolor=color, edgecolor=color, alpha=0.38, linewidth=2))

    label_positions = {
        "100": (-1.60, 0.70),
        "010": (1.60, 0.70),
        "001": (0.0, -1.78),
        "110": (0.0, 0.93),
        "101": (-0.70, -0.60),
        "011": (0.70, -0.60),
        "111": (0.0, 0.03),
    }
    for region, position in label_positions.items():
        axis.text(*position, str(counts[region]), ha="center", va="center", fontsize=14, fontweight="bold")

    set_label_positions = ((-1.55, 2.08), (1.55, 2.08), (0.0, -2.55))
    for run, position, color in zip(runs, set_label_positions, COLORS, strict=True):
        axis.text(
            *position,
            f"{textwrap.fill(run.label, width=26)}\ncorrect: {len(run.correct_ids)}",
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            fontweight="bold",
        )

    axis.set_title(
        f"TeleQnA: exact overlap of correct question IDs ({len(runs[0].all_ids)} questions)",
        fontsize=14,
        pad=20,
    )
    axis.text(
        0,
        -2.98,
        "Each number is an exclusive region; a question is correct only when is_correct (or legacy correct) is true.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    axis.set_aspect("equal")
    axis.set_xlim(-2.9, 2.9)
    axis.set_ylim(-3.2, 2.55)
    axis.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    runs = validate_runs((load_run(label, path) for label, path in args.run), args.expected_count)
    counts = region_counts(runs)
    output = args.output.expanduser()
    if output.suffix.lower() != ".png":
        raise ValueError("--output must end in .png")
    plot(runs, counts, output)

    summary = {
        "expected_question_count": args.expected_count,
        "runs": [
            {"label": run.label, "path": str(run.path), "correct_count": len(run.correct_ids)} for run in runs
        ],
        "exclusive_correct_regions": counts,
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {summary_path}")
    print("Exclusive correct regions:", json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
