"""Worker entry point: run one (problem, arm, seed) and write its raw evaluations as JSON.

Metrics are not computed here; the master derives hypervolume, feasible counts and times-to-target from the
stored evaluations, so every machine only needs numpy, scipy, scikit-learn and BoTorch.

  python bench_run.py --problem MW3 --arm FA-MOBO --seed 101 --out results/
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

from bench_arms import run_arm  # noqa: E402
from bench_problems import PROBLEMS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=sorted(PROBLEMS))
    ap.add_argument("--arm", required=True, choices=["FA-MOBO", "Aug-BO", "Clf-BO", "qNEHVI"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.problem}_{a.arm}_{a.seed}.json"
    if path.exists():
        print(f"[skip] {path.name}"); return
    t0 = time.time()
    rec = run_arm(PROBLEMS[a.problem], a.arm, a.seed)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["host"] = os.uname().nodename
    path.write_text(json.dumps(rec))
    feas = sum(rec["feasible"])
    print(f"[done] {path.name}: {len(rec['F'])} evaluations, {feas} feasible, {rec['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
