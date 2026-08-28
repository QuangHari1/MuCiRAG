"""Build reproducible document selections for Release 18 embedding runs.

``full-gsma-rel18`` selects every local GSMA ``raw.md``.  The
``paper-baseline-gsma-rel18`` selection maps each Telco-oRAG paper DOCX to its
version-equivalent GSMA Markdown file.  The output is metadata only; it never
downloads, chunks, or embeds text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from newbaseline.src.settings import load_settings
from newbaseline.src.corpus import discover_source_documents

SETTINGS = load_settings()
WORKSPACE_ROOT = SETTINGS.workspace_root
RELEASE_DIR = SETTINGS.release_dir
MAPPING_PATH = SETTINGS.dataset_dir / "3gpp/paper_gsma_rel18_mapping.json"
SELECTION_DIR = SETTINGS.dataset_dir / "3gpp/embedding_selections"
PAPER_SELECTION_ID = SETTINGS.get("paper", "paper_selection_id")
FULL_SELECTION_ID = SETTINGS.get("paper", "full_selection_id")
PAPER_VERSION_PATTERN = re.compile(r"(?P<variant>.+)-i\d{2}$")


def workspace_relative(path: Path) -> str:
    return path.relative_to(WORKSPACE_ROOT).as_posix()


def document_key_from_raw_path(raw_path: Path, release_dir: Path) -> str:
    """Return ``<series>_series/<document-directory>`` for any selected Markdown source."""
    relative = raw_path.relative_to(release_dir)
    if len(relative.parts) != 3:
        raise ValueError(f"Unexpected GSMA Release 18 path: {raw_path}")
    return relative.parent.as_posix()


def build_full_selection(release_dir: Path) -> dict[str, object]:
    sources = discover_source_documents(
        release_dir,
        (SETTINGS.get("chunking", "preferred_source_name"), SETTINGS.get("chunking", "fallback_source_name")),
    )
    documents = [
        {
            "document_key": source.document_key,
            "source_markdown_path": workspace_relative(source.path),
            "source_file_name": source.source_name,
            "selection_status": "full_corpus",
        }
        for source in sources
    ]
    if not documents:
        raise FileNotFoundError(f"No raw.md files found in {release_dir}")
    return {
        "selection_id": FULL_SELECTION_ID,
        "selection_mode": "full_gsma_release_18",
        "release": SETTINGS.release,
        "document_count": len(documents),
        "documents": documents,
    }


def exact_variant_paths(row: dict[str, object]) -> list[str]:
    filename = row.get("paper_filename")
    paths = row.get("gsma_raw_markdown_paths")
    if not isinstance(filename, str) or not isinstance(paths, list):
        raise ValueError(f"Malformed mapping row: {row}")

    stem = Path(filename).stem
    version_match = PAPER_VERSION_PATTERN.fullmatch(stem)
    if version_match is None:
        return []
    expected_directory = version_match.group("variant")
    return [
        path for path in paths
        if isinstance(path, str) and Path(path).parent.name == expected_directory
    ]


def build_paper_selection(
    mapping_path: Path, include_release_summaries: bool = False, release_dir: Path = RELEASE_DIR
) -> dict[str, object]:
    with mapping_path.open(encoding="utf-8") as mapping_file:
        mapping = json.load(mapping_file)
    rows = mapping.get("documents")
    if not isinstance(rows, list):
        raise ValueError(f"Missing documents list in {mapping_path}")

    available_sources = {
        source.document_key: source
        for source in discover_source_documents(
            release_dir,
            (SETTINGS.get("chunking", "preferred_source_name"), SETTINGS.get("chunking", "fallback_source_name")),
        )
    }
    documents: list[dict[str, str]] = []
    unresolved: list[str] = []
    selected_paths: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("kind") != "specification":
            continue
        filename = row.get("paper_filename")
        canonical_id = row.get("canonical_spec_id")
        candidates = row.get("gsma_raw_markdown_paths")
        if not isinstance(filename, str) or not isinstance(canonical_id, str) or not isinstance(candidates, list):
            raise ValueError(f"Malformed paper specification row: {row}")

        exact_paths = exact_variant_paths(row)
        if len(exact_paths) == 1:
            selected_path = exact_paths[0]
            status = "exact_variant_match"
        elif not exact_paths and len(candidates) == 1 and isinstance(candidates[0], str):
            selected_path = candidates[0]
            status = "single_candidate_fallback"
        else:
            unresolved.append(filename)
            continue

        if selected_path in selected_paths:
            raise ValueError(f"Multiple paper files selected the same GSMA document: {selected_path}")
        selected_paths.add(selected_path)
        mapped_path = Path(selected_path)
        if len(mapped_path.parts) < 3:
            raise ValueError(f"Unexpected mapped source path: {selected_path}")
        document_key = Path(mapped_path.parent.parent.name, mapped_path.parent.name).as_posix()
        source = available_sources.get(document_key)
        if source is None:
            unresolved.append(filename)
            continue
        documents.append(
            {
                "document_key": document_key,
                "source_markdown_path": workspace_relative(source.path),
                "source_file_name": source.source_name,
                "mapped_gsma_raw_markdown_path": selected_path,
                "paper_filename": filename,
                "canonical_spec_id": canonical_id,
                "selection_status": status,
            }
        )

    if include_release_summaries:
        # Paper-mode is intentionally 549 mapped specifications plus these
        # four DOCX summaries; the summaries have their own document keys and
        # are never substituted for a numbered GSMA specification.
        summary_dir = WORKSPACE_ROOT / SETTINGS.get("paper_release_summaries", "source_dir")
        for release in SETTINGS.get("paper_release_summaries", "source_releases"):
            filename = f"rel_{release}.docx"
            source_path = summary_dir / filename
            if not source_path.exists():
                raise FileNotFoundError(f"Missing paper release summary: {source_path}")
            documents.append(
                {
                    "document_key": f"paper_release_summaries/rel_{release}",
                    "paper_filename": filename,
                    "paper_source_path": workspace_relative(source_path),
                    "source_release": release,
                    "selection_status": "paper_release_summary",
                }
            )

    if unresolved:
        raise ValueError(
            "No unambiguous GSMA counterpart for paper files: " + ", ".join(unresolved)
        )
    return {
        "selection_id": PAPER_SELECTION_ID,
        "selection_mode": "telco_orag_paper_equivalent_gsma_release_18",
        "release": SETTINGS.release,
        "mapping_path": workspace_relative(mapping_path),
        "document_count": len(documents),
        "exact_variant_matches": sum(
            document["selection_status"] == "exact_variant_match" for document in documents
        ),
        "single_candidate_fallbacks": sum(
            document["selection_status"] == "single_candidate_fallback" for document in documents
        ),
        "release_summaries": sum(
            document["selection_status"] == "paper_release_summary" for document in documents
        ),
        "documents": documents,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
    ) as temporary_file:
        json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("paper", "full"), required=True)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--release-dir", type=Path, default=RELEASE_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--without-release-summaries",
        action="store_true",
        help="For paper mode only: select the 549 numbered specifications without the four paper summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "full":
        report = build_full_selection(args.release_dir)
        output_path = args.output or SELECTION_DIR / f"{FULL_SELECTION_ID}.json"
    else:
        report = build_paper_selection(
            args.mapping,
            include_release_summaries=not args.without_release_summaries,
            release_dir=args.release_dir,
        )
        output_path = args.output or SELECTION_DIR / f"{PAPER_SELECTION_ID}.json"
    write_json(output_path, report)
    print(f"Wrote {report['document_count']} selected documents to {output_path}")


if __name__ == "__main__":
    main()
