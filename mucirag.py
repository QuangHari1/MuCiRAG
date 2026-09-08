"""Run MuCiRAG commands from the repository root.

Examples:
    uv run --project MuCiRAG python mucirag.py ask "What is network slicing?"
    uv run --project MuCiRAG python mucirag.py benchmark --help
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


COMMANDS = {
    "ask": "MuCiRAG.src.rag.cli",
    "benchmark": "MuCiRAG.scripts.run_teleqna_benchmark",
    "summarize": "MuCiRAG.analysis.summarize_teleqna_runs",
    "assets": "MuCiRAG.scripts.manage_assets",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="MuCiRAG: telecom question answering and evaluation.")
    parser.add_argument("command", choices=COMMANDS, help="Use COMMAND --help for its options.")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # Keep dispatch free of provider/model imports so help works before setup.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.argv = [args.command, *args.arguments]
    runpy.run_module(COMMANDS[args.command], run_name="__main__")


if __name__ == "__main__":
    main()
