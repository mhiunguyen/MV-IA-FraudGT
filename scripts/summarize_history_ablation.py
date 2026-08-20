"""Summarize the seed-43 R/F/M history ablation without test-set tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = ("A", "H-R", "H-F", "H-M", "H-RFM")


def read_json_lines(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def locate_run_dir(results_root: Path, model: str) -> Path:
    pattern = f"AML-Small-HI-Ablation-{model}-Seed43*-gpu*"
    matches = sorted(path for path in results_root.glob(pattern) if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one run directory for {model} with {pattern}; "
            f"found {[str(path) for path in matches]}"
        )
    return matches[0]


def summarize_model(run_dir: Path, model: str, threshold: float) -> dict:
    seed_dir = run_dir / "43"
    val_records = read_json_lines(seed_dir / "val" / "stats.json")
    test_records = read_json_lines(seed_dir / "test" / "stats.json")
    threshold_percent = int(round(threshold * 100))
    suffix = f"{threshold_percent:02d}"
    f1_key = f"f1_t{suffix}"
    candidates = [record for record in val_records if f1_key in record]
    if not candidates:
        raise RuntimeError(f"{f1_key} not found in {seed_dir / 'val' / 'stats.json'}")

    # Epoch selection is based on validation F1 only.  Test metrics are read
    # exactly once at the selected epoch and never participate in selection.
    best_val = max(candidates, key=lambda record: float(record[f1_key]))
    epoch = int(best_val["epoch"])
    test = next(
        (record for record in test_records if int(record["epoch"]) == epoch),
        None,
    )
    if test is None:
        raise RuntimeError(f"No test record at epoch {epoch} in {seed_dir}")

    return {
        "model": model,
        "seed": 43,
        "best_epoch_by_validation": epoch,
        "threshold": threshold,
        "val_f1": float(best_val[f1_key]),
        "test_f1": float(test[f1_key]),
        "test_precision": float(test[f"precision_t{suffix}"]),
        "test_recall": float(test[f"recall_t{suffix}"]),
        "test_auc": float(test["auc"]),
        "parameters": int(test.get("params", 0)),
        "gpu_memory_mib": float(test.get("gpu_memory", 0)),
        "run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    rows = [
        summarize_model(locate_run_dir(args.results_root, model), model,
                        args.threshold)
        for model in MODELS
    ]
    frame = pd.DataFrame(rows)
    baseline_f1 = float(frame.loc[frame["model"] == "A", "test_f1"].iloc[0])
    frame["delta_f1_vs_A"] = frame["test_f1"] - baseline_f1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    display_columns = [
        "model", "best_epoch_by_validation", "threshold", "val_f1",
        "test_f1", "delta_f1_vs_A", "test_precision", "test_recall",
        "test_auc", "parameters", "gpu_memory_mib",
    ]
    print(frame[display_columns].to_string(index=False))
    print(f"\nCSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()
