"""Aggregate the benchmark runs: per-run metrics, per-problem summaries and paired tests against FA-MOBO.

  python bench_report.py targets    # fixed hypervolume targets from a 20,000-point random search per problem
  python bench_report.py report     # read results_all/*.json and write out/bench_numbers.json and the table

Analysis caveats, established from the first 123 runs before the study finished,
so the choices below were fixed without sight of the complete results.

* Every arm shares the seeding phase for a given seed. This was verified
  byte-identical: 50 evaluations on the five-variable MW problems, 60 on OSY.
  A target met at or before that boundary is therefore identical across arms by
  construction, and "evaluations to target" is diluted wherever the seed alone
  solves a run: MW11 20-33% of runs, MW3 50-67%, MW2 and MW7 none. The per-arm
  share_target_in_seed_phase field reports this and belongs beside the metric.
  TARGET_FRACTIONS reports the metric at several target levels for the same
  reason; 0.5 stays primary, and 0.9-0.95 is unreached by most arms.

* OSY cannot support an evaluations-to-target claim. With hv_ref (0, 386) one
  feasible point dominates the hypervolume, so a single evaluation carries the
  curve from roughly 12k-35k past both the 0.5 threshold (39.6k) and the 0.95
  one (75.3k). The crossing consequently lands on each arm's first model-driven
  evaluation: 70 for the augmented arms, 61 for qNEHVI, which has no
  augmentation phase. That 9-evaluation gap is phase structure, not search
  quality, and must not be read as qNEHVI reaching the target sooner. Report OSY
  final hypervolume and feasible count only.

* POST HOC, derived after the complete results were examined and not a predefined
  hypothesis: the augmentation stage behaves as a budget reallocation, and whether it
  pays is predicted by aug_evals * (augmentation conversion rate - BO conversion rate),
  where a conversion rate is feasible designs found per evaluation spent in that phase.
  Across the six problems this predictor tracks the measured paired difference against
  qNEHVI with r = 0.971 (MW11 predicted +1.88 against measured +2.50; OSY predicted
  -7.34 against measured -7.15). The sign agrees on five of six problems; the exception
  is MW2, where both the predicted and the measured values are indistinguishable from
  zero (p = 0.504), so it is noise about zero rather than a counterexample. Any write-up
  must present this as exploratory and label it as selected after examining results.

* THAT RULE WAS THEN TESTED AND PARTLY FALSIFIED. It was pre-registered against two unseen
  problems from a family it had never been fitted to (C2DTLZ2, C3DTLZ4; see
  pre_registration.json, written before any design-seed run and pinned by source hash) using
  rates estimated only from pilot seeds 198 and 199. It got one of two signs right. On
  C2DTLZ2 it predicted -3.36 and the measured paired difference was +4.50 in favour of
  FA-MOBO (won 15 of 19 decided seeds, Wilcoxon p = 0.0295). On C3DTLZ4 the sign held
  (+0.46 predicted, +3.15 measured, p = 0.0727) but the magnitude was out by a factor of
  seven. Across all eight problems the displacement rule degrades from r = 0.971 in sample
  to r = 0.867 with six of eight signs correct. Report it as a partly falsified heuristic,
  never as a validated law.

* WHAT THE RULE OMITS, and the better account. Treating augmentation as pure displacement
  ignores that the augmented data also trains a better feasibility surrogate, so the BO
  evaluations that follow convert at a higher rate. Decomposing the measured difference into
  a direct term, n_aug * (augmentation rate - control BO rate), and an indirect term,
  n_bo * (FA-MOBO BO rate - control BO rate), reproduces it exactly (r = 1.000, 8/8 signs),
  because both arms spend the same post-seed budget on a shared seed. The indirect term
  carries 44% of the explained magnitude and outweighs the direct term on four of the eight
  problems. C2DTLZ2 is the clearest case: direct -0.06, indirect +4.56. FA-MOBO's BO phase
  converts better than the control's on six of eight problems. This is a decomposition of
  measured quantities, not a predictive model, and it was also derived after the fact.

* THE CIRCUIT ON THE SAME AXIS (make_axis_table.py). The Class-E PA study splits its budget
  the same way, so the same decomposition applies to it: over the ten paired seeds the
  augmentation phase converts at 0.293 and the FA-MOBO BO phase at 0.364, against 0.169 for
  the plain qNEHVI control, with a seed feasibility of 0.020 that matches the ~2% the
  manuscript reports. Direct 15 * (0.293 - 0.169) = +1.86, indirect 66 * (0.364 - 0.169) =
  +12.84, summing to the measured +14.70 exactly. The indirect channel therefore carries 87%
  of the circuit result: the classifier trained on augmented data more than doubles the BO
  phase's feasible yield. The circuit shows the largest rate gap of any problem studied,
  which is why its advantage is far larger than any benchmark's.

* WHAT DOES NOT HOLD. "A low control BO conversion rate predicts an FA-MOBO win" is refuted
  by C2DTLZ2, where the control rate is high (0.641) and FA-MOBO still wins, entirely through
  the indirect term. The quantity that tracks the advantage is the gap between the FA-MOBO and
  control BO rates, not the level of the control rate. Both statements are post hoc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_problems import PROBLEMS  # noqa: E402

ARMS = ("FA-MOBO", "Aug-BO", "Clf-BO", "qNEHVI")
MAIN = "FA-MOBO"
TARGET_FRACTION = 0.5            # "evaluations to target" uses half the random-search reference hypervolume
TARGET_FRACTIONS = (0.5, 0.7, 0.9, 0.95)   # sweep reported beside the primary fraction: all arms share the
                                           # seeding phase, so an easy target can be met before any arm acts
RANDOM_SEARCH_N = 20_000
DESIGN_SEEDS = frozenset(range(101, 121))   # the study is exactly these 20 seeds per arm and problem
TARGETS = HERE / "bench_targets.json"


def hv2(F: np.ndarray, ref: tuple[float, float]) -> float:
    """Exact 2-D dominated hypervolume for minimization with respect to ref (points worse than ref are ignored)."""
    if len(F) == 0:
        return 0.0
    P = F[(F[:, 0] < ref[0]) & (F[:, 1] < ref[1])]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    keep, best = [], np.inf
    for row in P:                                  # keep the non-dominated staircase
        if row[1] < best:
            keep.append(row); best = row[1]
    P = np.array(keep)
    widths = np.diff(np.append(P[:, 0], ref[0]))
    heights = ref[1] - P[:, 1]
    return float(np.sum(widths * heights))


def make_targets() -> None:
    """Reference hypervolume per problem from a uniform random search, fixed before the study is analysed."""
    rng = np.random.default_rng(12345)
    out = {}
    for name, p in PROBLEMS.items():
        X = p.xl + rng.random((RANDOM_SEARCH_N, p.n_var)) * (p.xu - p.xl)
        F, G = p.evaluate(X)
        feas = (G <= 0).all(axis=1)
        h = hv2(F[feas], p.hv_ref)
        out[name] = {"random_search_hv": h, "target_hv": TARGET_FRACTION * h, "feasible_fraction": float(feas.mean()),
                     "n_samples": RANDOM_SEARCH_N, "hv_ref": list(p.hv_ref)}
        print(f"{name:5s} random-search HV {h:10.4f}  target {TARGET_FRACTION * h:10.4f}  feasible {feas.mean():6.2%}")
    TARGETS.write_text(json.dumps(out, indent=1))
    print("wrote", TARGETS.name)


def run_metrics(rec: dict, tgt: dict, ref: tuple[float, float]) -> dict:
    F = np.array(rec["F"], float)
    feas = np.array(rec["feasible"], bool)
    hv_curve, best = [], 0.0
    for i in range(len(F)):
        if feas[i]:
            best = max(best, hv2(F[: i + 1][feas[: i + 1]], ref))
        hv_curve.append(best)
    hv_curve = np.array(hv_curve)
    seed_evals = sum(1 for ph in rec.get("phase", []) if ph == "seed")

    def reach(frac: float):
        """First evaluation whose hypervolume meets `frac` of the random-search reference."""
        idx = np.flatnonzero(hv_curve >= frac * tgt["random_search_hv"])
        return int(idx[0]) + 1 if len(idx) else None

    primary = reach(TARGET_FRACTION)
    return {"problem": rec["problem"], "arm": rec["arm"], "seed": rec["seed"], "evaluations": len(F),
            "feasible": int(feas.sum()), "found_feasible": bool(feas.any()),
            "first_feasible": int(np.argmax(feas)) + 1 if feas.any() else None,
            "hv": float(hv_curve[-1]), "evals_to_target": primary,
            "evals_to_target_by_fraction": {str(f): reach(f) for f in TARGET_FRACTIONS},
            "seed_phase_evals": seed_evals,
            "target_reached_in_seed_phase": bool(primary is not None and primary <= seed_evals),
            "seconds": rec.get("seconds")}


def report() -> None:
    targets = json.loads(TARGETS.read_text())
    recs = [json.loads(f.read_text()) for f in sorted((HERE / "results_all").glob("*.json"))]
    if not recs:
        raise SystemExit("no results in results_all/")
    stray = [r for r in recs if r["seed"] not in DESIGN_SEEDS]
    if stray:                      # a run outside the design would shift one arm's mean on a different sample
        for r in stray:
            print(f"EXCLUDED, seed outside the design: {r['problem']} {r['arm']} seed {r['seed']}")
        recs = [r for r in recs if r["seed"] in DESIGN_SEEDS]
    counts: dict = {}
    for r in recs:
        counts[(r["problem"], r["arm"])] = counts.get((r["problem"], r["arm"]), 0) + 1
    over = {k: v for k, v in counts.items() if v > len(DESIGN_SEEDS)}
    if over:
        raise SystemExit(f"more runs than seeds for {over}; results_all/ holds duplicates")
    rows = [run_metrics(r, targets[r["problem"]], tuple(targets[r["problem"]]["hv_ref"])) for r in recs]
    per_arm: dict = {}
    for prob in sorted(PROBLEMS):
        per_arm[prob] = {}
        for arm in ARMS:
            sel = [r for r in rows if r["problem"] == prob and r["arm"] == arm]
            if not sel:
                continue
            hv = np.array([r["hv"] for r in sel]); fe = np.array([r["feasible"] for r in sel])
            tt = [r["evals_to_target"] for r in sel]
            per_arm[prob][arm] = {
                "runs": len(sel), "hv_mean": float(hv.mean()), "hv_sd": float(hv.std(ddof=1)) if len(hv) > 1 else 0.0,
                "feasible_mean": float(fe.mean()), "p_found_feasible": float(np.mean([r["found_feasible"] for r in sel])),
                "runs_reaching_target": int(sum(t is not None for t in tt)),
                "median_evals_to_target": float(np.median([t for t in tt if t is not None])) if any(t is not None for t in tt) else None,
                "median_first_feasible": float(np.median([r["first_feasible"] for r in sel if r["first_feasible"]])) if any(r["first_feasible"] for r in sel) else None,
            }
            for f in TARGET_FRACTIONS:
                got = [t for t in (r["evals_to_target_by_fraction"][str(f)] for r in sel) if t is not None]
                per_arm[prob][arm][f"reach_{f}"] = {"runs_reaching": len(got),
                                                    "median_evals": float(np.median(got)) if got else None}
            per_arm[prob][arm]["share_target_in_seed_phase"] = float(np.mean([r["target_reached_in_seed_phase"] for r in sel]))
    tests: dict = {}
    for prob in sorted(PROBLEMS):
        tests[prob] = {}
        for arm in ARMS:
            if arm == MAIN or arm not in per_arm.get(prob, {}):
                continue
            a = {r["seed"]: r for r in rows if r["problem"] == prob and r["arm"] == MAIN}
            b = {r["seed"]: r for r in rows if r["problem"] == prob and r["arm"] == arm}
            seeds = sorted(set(a) & set(b))
            for met in ("hv", "feasible"):
                d = np.array([a[s][met] - b[s][met] for s in seeds], float)
                nz = d[d != 0]
                tests[prob].setdefault(met, {})[arm] = {
                    "diff_mean": float(d.mean()), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                    "p": float(wilcoxon(nz).pvalue) if len(nz) else 1.0, "seeds": len(seeds)}
    out = HERE / "out"; out.mkdir(exist_ok=True)
    (out / "bench_numbers.json").write_text(json.dumps({"per_run": rows, "per_arm": per_arm, "tests": tests, "targets": targets}, indent=1))
    print(f"{'problem':6s} {'arm':9s} {'runs':>4s} {'HV mean':>9s} {'feasible':>8s} {'P(feas)':>7s} {'to target':>9s}")
    for prob in sorted(per_arm):
        for arm in ARMS:
            s = per_arm[prob].get(arm)
            if s:
                tt = f"{s['median_evals_to_target']:.0f}" if s["median_evals_to_target"] else "--"
                print(f"{prob:6s} {arm:9s} {s['runs']:4d} {s['hv_mean']:9.4f} {s['feasible_mean']:8.1f} {s['p_found_feasible']:7.0%} {tt:>9s}")
    print()
    print("evaluations to target by target fraction (-- = target never reached; seed% = reached during the shared seeding phase)")
    head = " ".join(f"{f:>9}" for f in TARGET_FRACTIONS)
    print(f"{'problem':6s} {'arm':9s} {head} {'seed%':>6s}")
    for prob in sorted(per_arm):
        for arm in ARMS:
            s_ = per_arm[prob].get(arm)
            if not s_:
                continue
            cells = []
            for f in TARGET_FRACTIONS:
                e = s_[f"reach_{f}"]
                cells.append(f"{e['median_evals']:.0f}({e['runs_reaching']})" if e["median_evals"] else "--")
            row = " ".join(f"{c:>9s}" for c in cells)
            print(f"{prob:6s} {arm:9s} {row} {s_['share_target_in_seed_phase']:6.0%}")
    print("wrote out/bench_numbers.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["targets", "report"])
    {"targets": make_targets, "report": report}[ap.parse_args().command]()
