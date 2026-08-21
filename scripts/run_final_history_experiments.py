"""Run the complete seed-43 H-FraudGT study with safe resume checkpoints.

The generated variants share every training hyperparameter. They differ only
in the selected history group(s), plus HG's explicitly declared reliability
gate. With two GPUs, phases are deterministic so an interrupted invocation can
be resumed by running the same command again.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


MODEL_SPECS = {
    "A": {"add_history": False, "groups": ["recency", "frequency", "monetary"]},
    "H-R": {"add_history": True, "groups": ["recency"]},
    "H-F": {"add_history": True, "groups": ["frequency"]},
    "H-M": {"add_history": True, "groups": ["monetary"]},
    "H-RF": {"add_history": True, "groups": ["recency", "frequency"]},
    "H-RM": {"add_history": True, "groups": ["recency", "monetary"]},
    "H-FM": {"add_history": True, "groups": ["frequency", "monetary"]},
    "H-RFM": {
        "add_history": True,
        "groups": ["recency", "frequency", "monetary"],
    },
    "HG": {
        "add_history": True,
        "groups": ["recency", "frequency", "monetary"],
        "reliability": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Generate the nine configs, validate the matrix, and exit",
    )
    return parser.parse_args()


def generated_config_dir(repo: Path) -> Path:
    return repo / "generated_configs_final_history"


def config_path(repo: Path, model: str) -> Path:
    return generated_config_dir(repo) / f"AML-Small-HI-Final-{model}-Seed43.yaml"


def write_configs(repo: Path) -> None:
    base_path = (
        repo / "configs" / "AML-Small-HI" /
        "AML-Small-HI-History-Final-Seed43.yaml"
    )
    with base_path.open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)

    output_dir = generated_config_dir(repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    for model, spec in MODEL_SPECS.items():
        cfg = copy.deepcopy(base)
        cfg["dataset"]["add_history"] = bool(spec["add_history"])
        cfg["dataset"]["history_groups"] = list(spec["groups"])
        cfg["dataset"]["history_reliability"] = bool(
            spec.get("reliability", False)
        )
        with config_path(repo, model).open("w", encoding="utf-8") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)


def result_dir(repo: Path, model: str, gpu: int) -> Path:
    stem = config_path(repo, model).stem
    return repo / "results_final_history" / f"{stem}-gpu{gpu}" / "43"


def is_complete(repo: Path, model: str, gpu: int) -> bool:
    run = result_dir(repo, model, gpu)
    stats_path = run / "test" / "stats.json"
    best_path = run / "ckpt" / "best.ckpt"
    if not stats_path.exists() or not best_path.exists():
        return False
    try:
        records = [
            json.loads(line) for line in stats_path.read_text(
                encoding="utf-8").splitlines() if line.strip()
        ]
        return any(int(record.get("epoch", -1)) == 99 for record in records)
    except (OSError, ValueError, TypeError):
        return False


def command(repo: Path, model: str, gpu: int) -> list[str]:
    return [
        sys.executable, "-u", "-m", "fraudGT.main",
        "--cfg", str(config_path(repo, model)),
        "--repeat", "1", "--gpu", str(gpu),
    ]


def run_phase(repo: Path, jobs: list[tuple[str, int]], poll_seconds: int) -> None:
    handles = []
    log_dir = Path("/kaggle/working/final_history_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    for model, gpu in jobs:
        if is_complete(repo, model, gpu):
            print(f"[skip] {model}: result and best.ckpt are complete", flush=True)
            continue
        log_path = log_dir / f"final_{model}_seed43.log"
        # Append so recovery attempts remain visible in one evidence file.
        stream = log_path.open("a", encoding="utf-8")
        stream.write(f"\n===== INVOCATION {time.ctime()} | GPU {gpu} =====\n")
        stream.flush()
        process = subprocess.Popen(
            command(repo, model, gpu), cwd=repo,
            stdout=stream, stderr=subprocess.STDOUT, text=True,
        )
        handles.append((model, gpu, log_path, process, stream))
        print(f"[start/resume] {model} on GPU {gpu}; PID={process.pid}", flush=True)

    started = time.time()
    while any(process.poll() is None for _, _, _, process, _ in handles):
        time.sleep(max(10, poll_seconds))
        states = ", ".join(
            f"{model}={'running' if process.poll() is None else 'done'}"
            for model, _, _, process, _ in handles
        )
        print(
            f"[heartbeat] {(time.time() - started) / 60:.0f} min | {states}",
            flush=True,
        )
        subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader"], check=False,
        )

    failures = []
    for model, gpu, log_path, process, stream in handles:
        stream.close()
        if process.returncode != 0 or not is_complete(repo, model, gpu):
            failures.append(model)
            lines = log_path.read_text(
                encoding="utf-8", errors="replace").splitlines()
            print(f"\n===== {model} INCOMPLETE: LOG TAIL =====")
            print("\n".join(lines[-120:]))
    if failures:
        raise RuntimeError(
            "Incomplete jobs (rerun the same command to resume): "
            + ", ".join(failures)
        )


def deterministic_phases(models: list[str], gpus: list[int]):
    requested = [model for model in MODEL_SPECS if model in set(models)]
    if len(gpus) == 1:
        return [[(model, gpus[0])] for model in requested]

    # Exactly one ordinary history variant must create data_history.pt before
    # the other R/F/M variants start. This avoids two processes writing the
    # same canonical cache when users split the experiment across accounts.
    ordinary_history = [
        model for model in requested
        if MODEL_SPECS[model]["add_history"]
        and not MODEL_SPECS[model].get("reliability", False)
    ]
    initializer = None
    if ordinary_history:
        initializer = "H-RFM" if "H-RFM" in ordinary_history \
            else ordinary_history[0]

    first_phase_models = []
    if initializer is not None:
        first_phase_models.append(initializer)
    # A and HG use different cache filenames, so either can safely run next
    # to the canonical-history initializer.
    for candidate in ("A", "HG"):
        if candidate in requested and candidate not in first_phase_models:
            first_phase_models.append(candidate)
            break
    if not first_phase_models and requested:
        first_phase_models.append(requested[0])

    remaining = [model for model in requested if model not in first_phase_models]
    phases = [[
        (model, gpus[index])
        for index, model in enumerate(first_phase_models[:2])
    ]]
    for start in range(0, len(remaining), 2):
        phases.append([
            (model, gpus[index])
            for index, model in enumerate(remaining[start:start + 2])
        ])
    return [phase for phase in phases if phase]


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    if not args.gpus:
        raise ValueError("At least one GPU index is required")
    write_configs(repo)
    print("Generated configs:", generated_config_dir(repo), flush=True)
    if args.prepare_only:
        return
    for index, phase in enumerate(
        deterministic_phases(args.models, args.gpus), start=1
    ):
        print(f"\n===== PHASE {index}: {phase} =====", flush=True)
        run_phase(repo, phase, args.poll_seconds)
    print("\nAll requested final history experiments completed.", flush=True)


if __name__ == "__main__":
    main()
