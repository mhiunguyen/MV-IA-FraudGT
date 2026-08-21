"""Summarize final history experiments and audit their best checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from run_final_history_experiments import MODEL_SPECS


def read_json_lines(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def locate_run_dir(results_root: Path, model: str) -> Path:
    pattern = f"AML-Small-HI-Final-{model}-Seed43-gpu*"
    matches = sorted(path for path in results_root.glob(pattern) if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one final run for {model}; found: "
            f"{[str(path) for path in matches]}"
        )
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_model(run_dir: Path, model: str) -> dict:
    seed_dir = run_dir / "43"
    val_records = read_json_lines(seed_dir / "val" / "stats.json")
    test_records = read_json_lines(seed_dir / "test" / "stats.json")
    candidates = [record for record in val_records if "f1_t50" in record]
    if not candidates:
        raise RuntimeError(f"f1_t50 missing from {seed_dir / 'val/stats.json'}")
    # Stable max chooses the earliest epoch in a tie, matching training.
    best_val = max(candidates, key=lambda record: float(record["f1_t50"]))
    epoch = int(best_val["epoch"])
    test = next(
        (record for record in test_records if int(record["epoch"]) == epoch),
        None,
    )
    if test is None:
        raise RuntimeError(f"No test record for selected epoch {epoch}")

    checkpoint = seed_dir / "ckpt" / "best.ckpt"
    if not checkpoint.exists():
        raise RuntimeError(f"Missing best checkpoint: {checkpoint}")
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(metadata.get("epoch", -1))
    checkpoint_metric = str(metadata.get("metric_name", ""))
    checkpoint_value = float(metadata.get("metric_value", float("nan")))
    checkpoint_ok = (
        checkpoint_epoch == epoch
        and checkpoint_metric == "f1_t50"
        and abs(checkpoint_value - float(best_val["f1_t50"])) < 1e-8
    )
    if not checkpoint_ok:
        raise RuntimeError(
            f"Checkpoint/validation mismatch for {model}: "
            f"ckpt(epoch={checkpoint_epoch}, metric={checkpoint_metric}, "
            f"value={checkpoint_value}) vs val(epoch={epoch}, "
            f"f1_t50={best_val['f1_t50']})"
        )

    return {
        "model": model,
        "seed": 43,
        "best_epoch_by_validation": epoch,
        "threshold": 0.50,
        "val_f1": float(best_val["f1_t50"]),
        "test_f1": float(test["f1_t50"]),
        "test_precision": float(test["precision_t50"]),
        "test_recall": float(test["recall_t50"]),
        "test_auc": float(test["auc"]),
        "test_accuracy": float(test.get("accuracy", float("nan"))),
        "parameters": int(test.get("params", 0)),
        "gpu_memory_mib": float(test.get("gpu_memory", 0)),
        "checkpoint_ok": checkpoint_ok,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    args = parser.parse_args()

    rows = [
        summarize_model(locate_run_dir(args.results_root, model), model)
        for model in args.models
    ]
    frame = pd.DataFrame(rows)
    baseline = float(frame.loc[frame["model"] == "A", "test_f1"].iloc[0]) \
        if "A" in set(frame["model"]) else float("nan")
    frame["delta_f1_vs_A"] = frame["test_f1"] - baseline
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    columns = [
        "model", "best_epoch_by_validation", "val_f1", "test_f1",
        "delta_f1_vs_A", "test_precision", "test_recall", "test_auc",
        "parameters", "gpu_memory_mib", "checkpoint_ok",
    ]
    print(frame[columns].to_string(index=False))
    print(f"\nCSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()
