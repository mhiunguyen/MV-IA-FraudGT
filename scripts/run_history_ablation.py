"""Run the seed-43 A/H-R/H-F/H-M/H-RFM ablation on one or two GPUs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


MODELS = ("A", "H-R", "H-F", "H-M", "H-RFM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--force", action="store_true",
                        help="Rerun even when a completed test log exists")
    return parser.parse_args()


def result_dir(repo: Path, model: str, gpu: int) -> Path:
    stem = f"AML-Small-HI-Ablation-{model}-Seed43"
    return repo / "results" / f"{stem}-gpu{gpu}"


def is_complete(repo: Path, model: str, gpu: int) -> bool:
    return (result_dir(repo, model, gpu) / "43" / "test" / "stats.json").exists()


def command(repo: Path, model: str, gpu: int) -> list[str]:
    cfg = repo / "configs" / "AML-Small-HI" / (
        f"AML-Small-HI-Ablation-{model}-Seed43.yaml"
    )
    return [
        sys.executable, "-u", "-m", "fraudGT.main", "--cfg", str(cfg),
        "--repeat", "1", "--gpu", str(gpu),
    ]


def run_phase(repo: Path, jobs: list[tuple[str, int]], force: bool) -> None:
    handles = []
    for model, gpu in jobs:
        if not force and is_complete(repo, model, gpu):
            print(f"[skip] {model}: completed result already exists", flush=True)
            continue
        log_path = Path("/kaggle/working") / f"ablation_{model}_seed43.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command(repo, model, gpu), cwd=repo, stdout=stream,
            stderr=subprocess.STDOUT, text=True,
        )
        handles.append((model, gpu, log_path, process, stream))
        print(f"[start] {model} on GPU {gpu}; PID={process.pid}", flush=True)

    started = time.time()
    while any(process.poll() is None for _, _, _, process, _ in handles):
        time.sleep(60)
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
    for model, _, log_path, process, stream in handles:
        stream.close()
        if process.returncode != 0:
            failures.append(model)
            lines = log_path.read_text(
                encoding="utf-8", errors="replace").splitlines()
            print(f"\n===== {model} FAILED: LOG TAIL =====")
            print("\n".join(lines[-100:]))
    if failures:
        raise RuntimeError("Failed ablation jobs: " + ", ".join(failures))


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    gpus = args.gpus
    if not gpus:
        raise ValueError("At least one GPU index is required")

    if len(gpus) == 1:
        phases = [[(model, gpus[0])] for model in MODELS]
    else:
        # H-RFM creates/validates the canonical full-history cache.  Partial
        # ablations run only after that phase, so they safely reuse the cache.
        phases = [
            [("A", gpus[0]), ("H-RFM", gpus[1])],
            [("H-F", gpus[0]), ("H-R", gpus[1])],
            [("H-M", gpus[0])],
        ]

    for index, phase in enumerate(phases, start=1):
        print(f"\n===== PHASE {index}/{len(phases)}: {phase} =====", flush=True)
        run_phase(repo, phase, args.force)
    print("\nAll seed-43 history ablation jobs completed.", flush=True)


if __name__ == "__main__":
    main()
