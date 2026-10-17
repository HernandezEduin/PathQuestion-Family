#!/usr/bin/env python3
"""Run the supplied PQ/PQL converter against the pinned original IRN submodule."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "IRN" / "PathQuestion"
OUTPUT = ROOT / "generated"
EXPECTED_ARCHIVE_MEMBER = "FB2M_TRI_NEW.txt"


def prepare_pql_source(source_dir: Path, output_dir: Path) -> Path:
    """Extract the upstream PQL text locally, rejecting unexpected ZIP paths."""
    archive = source_dir / "FB2M_TRI_NEW.txt.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Missing PQL upstream source archive: {archive}")
    destination = output_dir / EXPECTED_ARCHIVE_MEMBER
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = [entry for entry in bundle.infolist()
                   if not entry.is_dir() and Path(entry.filename).name == EXPECTED_ARCHIVE_MEMBER]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one {EXPECTED_ARCHIVE_MEMBER} member in {archive}")
        with bundle.open(members[0]) as src, destination.open("wb") as dst:
            import shutil
            shutil.copyfileobj(src, dst)
    return destination


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pq", "pql", "all"), default="all")
    parser.add_argument("--graph", choices=("filtered", "source", "both"), default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--encode-ids", action="store_true")
    parser.add_argument("--allow-source-coverage-gaps", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.split_ratio < 1.0:
        parser.error("--split-ratio must be strictly between 0 and 1")
    if not (UPSTREAM / "PQ-2H.txt").is_file():
        parser.error("Missing original data. Run: git submodule update --init --recursive")

    manifest_dir = OUTPUT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for dataset in (("pq", "pql") if args.dataset == "all" else (args.dataset,)):
        qa_dir = OUTPUT / "qa" / dataset.upper()
        triplet_dir = OUTPUT / "link_prediction" / dataset.upper()
        qa_dir.mkdir(parents=True, exist_ok=True)
        triplet_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable, str(ROOT / "scripts" / "pathqa_2_kgqa.py"),
            "--source-path", str(UPSTREAM),
            # The underlying converter appends /PQ or /PQL itself.
            # Pass the shared output roots to avoid PQ/PQ and PQL/PQL nesting.
            "--question-path", str(OUTPUT / "qa"),
            "--triplet-path", str(OUTPUT / "link_prediction"),
            "--graph-regime", args.graph,
            "--random-seed", str(args.seed),
            "--split-ratio", str(args.split_ratio),
        ]
        if dataset == "pql":
            command.append("--use-large")
            if args.graph in ("source", "both"):
                extracted = prepare_pql_source(UPSTREAM, OUTPUT / "source")
                command.extend(["--source-kg-file", str(extracted)])
        if args.encode_ids:
            command.append("--encode-ids")
        if args.allow_source_coverage_gaps:
            command.append("--allow-source-coverage-gaps")
        print(f"Reconstructing {dataset.upper()} ({args.graph}); seed={args.seed}", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)

        files = sorted([p for directory in (qa_dir, triplet_dir)
                        for p in directory.rglob("*") if p.is_file()])
        try:
            upstream_commit = subprocess.check_output(
                ["git", "-C", str(UPSTREAM.parent), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            upstream_commit = "unavailable"
        manifest = {
            "dataset": dataset.upper(),
            "graph_regime": args.graph,
            "seed": args.seed,
            "split_ratio": args.split_ratio,
            "encode_ids": args.encode_ids,
            "allow_source_coverage_gaps": args.allow_source_coverage_gaps,
            "python_version": sys.version.split()[0],
            "upstream_git_commit": upstream_commit,
            "outputs_sha256": {
                str(path.relative_to(OUTPUT)): sha256_file(path) for path in files
            },
        }
        (manifest_dir / f"{dataset}_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
