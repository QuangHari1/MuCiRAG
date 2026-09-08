"""Validate, stage, or download a checksum-verified MuCiRAG resource bundle.

Staging is offline. Upload the resulting directory with `hf upload` after
choosing a destination and visibility. No source, credentials, or results
are collected by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from fnmatch import fnmatchcase
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BUNDLE_MANIFEST = "assets-manifest.json"
RESOURCE_LOCK = PROJECT_ROOT / "MuCiRAG/resources.lock.json"
# Explicit roots/extensions keep caches, secrets, experiments and partial
# embedding checkpoints out of the release.
ASSET_ROOTS = {
    "dataset/3gpp/Chunk/Rel-18": {".json"},
    "dataset/3gpp/Embeddings/Rel-18/paper-baseline-gsma-rel18": {
        ".json", ".jsonl", ".npy", ".npz", ".sqlite3",
    },
}
SKIP_PARTS = {".cache", ".partial", "__pycache__"}
ASSET_FILES = [
    "dataset/teleqna/TeleQnA.json",
    "dataset/3gpp/embedding_selections/paper-baseline-gsma-rel18.json",
    "MuCiRAG/resources/router_new.pth",
    "MuCiRAG/resources/series_description.json",
    "MuCiRAG/resources/3GPP_vocabulary.docx",
    "MuCiRAG/resources/3GPP_vocabulary_release18.json",
    "MuCiRAG/resources/3GPP_definitions_paper_v17.json",
]


def is_runtime_asset(relative: str) -> bool:
    """Only bundle data paths can be restored into a source checkout."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or SKIP_PARTS.intersection(path.parts):
        return False
    return relative in ASSET_FILES or any(
        fnmatchcase(relative, directory + "/*") and path.suffix in extensions
        for directory, extensions in ASSET_ROOTS.items()
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_files(root: Path) -> list[Path]:
    files = []
    for relative in ASSET_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or invalid resource file: {path}")
        files.append(path)
    for relative, extensions in ASSET_ROOTS.items():
        directory = root / relative
        if not directory.is_dir() or directory.is_symlink():
            raise FileNotFoundError(f"Missing resource directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Resource symlinks are not supported: {path}")
            if path.is_file() and path.suffix in extensions and not SKIP_PARTS.intersection(path.parts):
                files.append(path)
    return files


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Resource path escapes bundle: {relative}")
    return path


def validate_corpus(root: Path) -> dict:
    """Check vector/metadata/chunk alignment and anchor provenance offline."""
    import numpy as np

    from MuCiRAG.src.rag.anchor_hierarchy import AnchorHierarchy
    from MuCiRAG.src.settings import load_settings

    settings = load_settings()
    embedding_root = safe_path(
        root, str(Path(settings.get("paths", "embedding_dir")) / settings.get("rag", "selection_id"))
    )
    manifest = json.loads((embedding_root / "manifest.json").read_text())
    chunk_root = safe_path(root, manifest["source_chunk_directory"])
    count = 0
    for series, record in manifest["series"].items():
        vector_path = safe_path(embedding_root, record["vector_file"])
        metadata_path = safe_path(embedding_root, record["metadata_file"])
        chunk_path = safe_path(chunk_root, record["chunk_file"])
        if sha256(metadata_path) != record["metadata_sha256"]:
            raise ValueError(f"Metadata checksum mismatch: {series}")
        vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
        metadata = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
        chunks = json.loads(chunk_path.read_text())["chunks"]
        expected_shape = (record["chunk_count"], manifest["dimensions"])
        if vectors.shape != expected_shape or len(metadata) != len(vectors):
            raise ValueError(f"Vector/metadata dimensions mismatch: {series}")
        for index, row in enumerate(metadata):
            source_index = row["source_chunk_index"]
            if not 0 <= source_index < len(chunks):
                raise ValueError(f"Invalid source chunk index: {series}:{index}")
            if row["embedding_row"] != index or chunks[source_index]["chunk_id"] != row["chunk_id"]:
                raise ValueError(f"Chunk identity mismatch: {series}:{index}")
        selected_chunks = [chunks[row["source_chunk_index"]] for row in metadata]
        source_digest = hashlib.sha256(json.dumps(
            selected_chunks, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if source_digest != record["source_sha256"]:
            raise ValueError(f"Selected chunk checksum mismatch: {series}")
        # Bound temporary allocations even for the largest series.
        for start in range(0, len(vectors), 4096):
            if not np.isfinite(vectors[start:start + 4096]).all():
                raise ValueError(f"Non-finite embedding values: {series}")
        count += len(vectors)
    anchor_root = safe_path(root, settings.get("rag", "anchor_hierarchy_embedding_root"))
    AnchorHierarchy.load(
        anchor_root,
        manifest_file=settings.get("rag", "anchor_hierarchy_manifest_file"),
        vectors_file=settings.get("rag", "anchor_hierarchy_vectors_file"),
        corpus_manifest_sha256=sha256(anchor_root / "manifest.json"),
        embedding_backend=settings.get("embedding", "backend"),
        embedding_model=settings.get("embedding", "model"),
        dimensions=settings.get("embedding", "dimensions"),
    )
    # The hybrid runtime must be usable without rebuilding the lexical index.
    import sqlite3

    lexical_path = embedding_root / settings.get("rag", "lexical_index_file")
    with sqlite3.connect(lexical_path.resolve().as_uri() + "?mode=ro", uri=True) as database:
        if database.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Lexical index failed SQLite integrity validation")
    for relative in ASSET_FILES:
        if not safe_path(root, relative).is_file():
            raise FileNotFoundError(f"Missing runtime resource: {relative}")
    return {"series": len(manifest["series"]), "vectors": count, "dimensions": manifest["dimensions"]}


def stage(root: Path, destination: Path) -> None:
    """Copy only release assets and record every file's size and SHA-256."""
    if destination.exists():
        raise FileExistsError(f"Choose a new staging directory: {destination}")
    stats = validate_corpus(root)
    files = asset_files(root)
    destination.mkdir(parents=True)
    records = []
    for index, source in enumerate(files, 1):
        relative = source.relative_to(root).as_posix()
        target = safe_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({"path": relative, "size": target.stat().st_size, "sha256": sha256(target)})
        if index % 100 == 0:
            print(f"Staged {index}/{len(files)} files", flush=True)
    manifest = {"format_version": 1, "project": "MuCiRAG", "corpus": stats, "files": records}
    (destination / BUNDLE_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(PROJECT_ROOT / "docs/huggingface-dataset-card.md", destination / "README.md")
    print(json.dumps({**stats, "files": len(records), "bytes": sum(row["size"] for row in records)}))


def verify(root: Path) -> None:
    manifest = json.loads((root / BUNDLE_MANIFEST).read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported resource manifest format")
    for row in manifest["files"]:
        path = safe_path(root, row["path"])
        if path.stat().st_size != row["size"] or sha256(path) != row["sha256"]:
            raise ValueError(f"Resource checksum mismatch: {row['path']}")
    print(f"Verified {len(manifest['files'])} resource files.")


def download(repo_id: str, revision: str, destination: Path, manifest_sha256: str | None = None) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    manifest_path = Path(hf_hub_download(repo_id, BUNDLE_MANIFEST, repo_type="dataset", revision=revision))
    if manifest_sha256 is not None and sha256(manifest_path) != manifest_sha256:
        raise ValueError("Remote bundle manifest differs from the pinned release")
    manifest = json.loads(manifest_path.read_text())
    paths = [row["path"] for row in manifest["files"]]
    if not paths or len(set(paths)) != len(paths) or not all(is_runtime_asset(path) for path in paths):
        raise ValueError("Remote manifest contains duplicate or unsupported runtime paths")
    for relative in paths:
        safe_path(destination, relative)

    snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision=revision, local_dir=destination,
        allow_patterns=paths,
    )
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, destination / BUNDLE_MANIFEST)
    verify(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate_parser = commands.add_parser("validate", help="Check the active local corpus offline")
    validate_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    stage_parser = commands.add_parser("stage", help="Prepare an upload directory offline")
    stage_parser.add_argument("--output", required=True, type=Path)
    verify_parser = commands.add_parser("verify", help="Verify a downloaded or staged bundle")
    verify_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    download_parser = commands.add_parser("download", help="Download resources from a pinned Hub revision")
    download_parser.add_argument("--repo", help="Override the repository pinned in resources.lock.json")
    download_parser.add_argument("--revision", help="Required with --repo; use an immutable commit SHA")
    download_parser.add_argument("--output", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(validate_corpus(args.root.resolve())))
    elif args.command == "stage":
        stage(PROJECT_ROOT, args.output.resolve())
    elif args.command == "verify":
        verify(args.root.resolve())
    else:
        if bool(args.repo) != bool(args.revision):
            parser.error("Provide both --repo and --revision, or neither to use resources.lock.json")
        if args.repo:
            download(args.repo, args.revision, args.output.resolve())
        else:
            release = json.loads(RESOURCE_LOCK.read_text())
            download(release["repo_id"], release["revision"], args.output.resolve(), release["manifest_sha256"])


if __name__ == "__main__":
    main()
