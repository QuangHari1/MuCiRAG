"""Build the persistent SQLite FTS5 index used by hybrid retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from newbaseline.src.rag.corpus import PaperEmbeddingCorpus
from newbaseline.src.settings import load_settings


def parse_args(default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {args.output}. Use --overwrite to rebuild it.")
    return args


def main() -> None:
    settings = load_settings()
    embedding_root = (
        settings.dataset_dir
        / "3gpp"
        / "Embeddings"
        / f"Rel-{settings.release}"
        / settings.get("rag", "selection_id")
    )
    default_output = embedding_root / settings.get("rag", "lexical_index_file")
    args = parse_args(default_output)
    if args.overwrite:
        args.output.unlink(missing_ok=True)
    corpus = PaperEmbeddingCorpus(embedding_root, settings.workspace_root)
    print(f"Building lexical index: {args.output}", flush=True)
    row_count = corpus.build_lexical_index(args.output)
    print(f"Done: {row_count} chunks indexed", flush=True)


if __name__ == "__main__":
    main()
