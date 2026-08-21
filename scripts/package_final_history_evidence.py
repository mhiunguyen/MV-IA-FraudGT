"""Create a reproducibility ZIP containing source, configs, logs and weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".ipynb"}
EXCLUDED_PARTS = {
    ".git", "__pycache__", "data", "outputs", "results",
    "results_final_history", "final_evidence_AH", "tmp",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_source(repo: Path, bundle: Path) -> None:
    source_root = bundle / "source"
    for name in ("LICENSE", ".gitignore"):
        path = repo / name
        if path.exists():
            copy_file(path, source_root / name)
    for path in repo.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(repo)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        copy_file(path, source_root / relative)


def copy_results(results_root: Path, bundle: Path) -> None:
    destination_root = bundle / "trained_runs"
    runs = sorted(results_root.glob("AML-Small-HI-Final-*-Seed43-gpu*"))
    if not runs:
        raise RuntimeError(f"No final runs found under {results_root}")
    for run in runs:
        seed_dir = run / "43"
        required = [
            seed_dir / "train" / "stats.json",
            seed_dir / "val" / "stats.json",
            seed_dir / "test" / "stats.json",
            seed_dir / "ckpt" / "best.ckpt",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Incomplete run {run.name}: {missing}")
        for path in required:
            copy_file(path, destination_root / run.name / path.relative_to(run))
        for path in (seed_dir / "ckpt").glob("[0-9]*.ckpt"):
            copy_file(path, destination_root / run.name / path.relative_to(run))
        dumped_config = run / "config.yaml"
        if dumped_config.exists():
            copy_file(
                dumped_config, destination_root / run.name / "config.yaml")


def write_manifest(bundle: Path) -> None:
    rows = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "SHA256_MANIFEST.json":
            rows.append({
                "path": path.relative_to(bundle).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    (bundle / "SHA256_MANIFEST.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    args = parser.parse_args()

    bundle = args.output_base.parent / (args.output_base.name + "_contents")
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    copy_source(args.repo.resolve(), bundle)
    copy_results(args.results_root.resolve(), bundle)

    copy_file(args.summary.resolve(), bundle / "reports" / args.summary.name)
    if args.logs_dir.exists():
        for log in args.logs_dir.glob("*.log"):
            copy_file(log, bundle / "logs" / log.name)
    if args.evidence_dir.exists():
        for path in args.evidence_dir.iterdir():
            if path.is_file():
                copy_file(path, bundle / "environment_and_data" / path.name)
    write_manifest(bundle)
    archive = shutil.make_archive(str(args.output_base), "zip", bundle)
    print(f"Evidence ZIP: {archive}")
    print(f"Bundle directory: {bundle}")


if __name__ == "__main__":
    main()
