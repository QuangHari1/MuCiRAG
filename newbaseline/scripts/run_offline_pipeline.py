"""Prepare one reproducible corpus: selection -> headings -> chunks -> optional embeddings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEWBASELINE_ROOT = PROJECT_ROOT / "newbaseline"
SCRIPTS_DIR = NEWBASELINE_ROOT / "scripts"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=NEWBASELINE_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("paper", "full"), required=True)
    parser.add_argument("--selection", type=Path, help="Reuse an existing selection JSON instead of regenerating it.")
    parser.add_argument("--embed", action="store_true", help="Run paid embedding calls after chunking.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "With --embed, rebuild headings/chunks then validate embedding inputs without calling the provider. "
            "Use embed_chunks.py directly to inspect an already-built corpus."
        ),
    )
    parser.add_argument("--series", action="append", help="Pass one or more numeric series to the embedding stage.")
    args = parser.parse_args()
    if args.dry_run and not args.embed:
        parser.error("--dry-run requires --embed")
    return args


def main() -> None:
    args = parse_args()
    selection = args.selection
    if selection is None:
        selection = PROJECT_ROOT / "dataset" / "3gpp" / "embedding_selections" / (
            "paper-baseline-gsma-rel18.json" if args.mode == "paper" else "full-gsma-rel18.json"
        )
        run([sys.executable, str(SCRIPTS_DIR / "prepare_embedding_selection.py"), "--mode", args.mode])

    run([sys.executable, str(SCRIPTS_DIR / "extract_heading.py"), "--selection", str(selection)])
    run([sys.executable, str(SCRIPTS_DIR / "chunking.py"), "--selection", str(selection)])
    if args.mode == "paper":
        run([sys.executable, str(SCRIPTS_DIR / "chunk_paper_release_summaries.py")])
    if not args.embed:
        print("Chunking complete. Re-run with --embed to create vectors.")
        return

    command = [
        sys.executable,
        str(SCRIPTS_DIR / "embed_chunks.py"),
        "--selection",
        str(selection),
    ]
    for series in args.series or []:
        command.extend(("--series", series))
    if args.dry_run:
        command.append("--dry-run")
    run(command)


if __name__ == "__main__":
    main()
