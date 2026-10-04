"""Numbers for the reliability additions, recomputed from the run histories.

Joint-target threshold sensitivity, simulations saved, worst-metric relative score,
high-margin yield of the augmentation and BO phases, and the steady feasible rate.
Writes out/novel_numbers.json.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest, wilcoxon

from common import ARMS, BUDGET, N_SEED, OUT, SEEDS, load_hist, margin_rich, per_seed_table
from problem import feasible_mask  # noqa: E402

MAIN = "fa15qc"
CAP = BUDGET + 1                      # unreached runs are counted at 182
N_BOOT, BOOT_SEED = 10_000, 0
F_GRID = list(range(6, 31, 2))        # feasible-count thresholds of the sensitivity grid
M_GRID = (1, 2)                       # new high-margin thresholds
X_GRID = (0.05, 0.10, 0.25)           # "within X of the best arm" tolerances
WINDOW = (116, 181)                   # simulations where every arm is in its BO phase
DESIGN_COLS = ["Lx_id", "Ldc3_id", "M3_width", "CB", "Cs", "Cm"]
H = {(a, s): load_hist(a, s) for a in ARMS for s in SEEDS}


def reach(h: pd.DataFrame, f: int, m: int) -> float:
    """First sim_index with cumulative strictly feasible >= f and cumulative new high-margin >= m (NaN if never)."""
    cf = np.cumsum(feasible_mask(h).to_numpy())
    cm = np.cumsum((margin_rich(h) & (h["sim_index"] > N_SEED)).to_numpy())
    hit = np.flatnonzero((cf >= f) & (cm >= m))
    return float(h["sim_index"].iloc[hit[0]]) if len(hit) else np.nan


def perm_p(d: np.ndarray) -> float:
    """Exact two-sided paired sign-flip permutation p of the mean difference (2^n relabelings)."""
    d = np.asarray(d, float)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12))


def boot_ci(d: np.ndarray) -> list[float]:
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(d), size=(N_BOOT, len(d)))
    lo, hi = np.percentile(np.asarray(d, float)[idx].mean(axis=1), [2.5, 97.5])
    return [float(lo), float(hi)]


def sign_test(ta: np.ndarray, tb: np.ndarray) -> tuple[int, int, float]:
    a, b = np.nan_to_num(ta, nan=np.inf), np.nan_to_num(tb, nan=np.inf)
    k, later = int((a < b).sum()), int((a > b).sum())
    return k, later, 1.0 if k + later == 0 else float(binomtest(k, k + later, 0.5).pvalue)


def time_block(f: int, m: int) -> dict:
    t = {a: np.array([reach(H[a, s], f, m) for s in SEEDS]) for a in ARMS}
    out = {"reached": {a: int(np.isfinite(t[a]).sum()) for a in ARMS},
           "capped_mean_beyond_seed": {a: float(np.nan_to_num(t[a], nan=CAP).mean() - N_SEED) for a in ARMS},
           "median_reached": {a: (float(np.nanmedian(t[a])) if np.isfinite(t[a]).any() else None) for a in ARMS}, "vs": {}}
    for a in ARMS:
        if a == MAIN:
            continue
        d = np.nan_to_num(t[a], nan=CAP) - np.nan_to_num(t[MAIN], nan=CAP)
        k, later, p = sign_test(t[MAIN], t[a])
        out["vs"][a] = {"saved_mean": float(d.mean()), "ci95": boot_ci(d), "sooner": k, "later": later, "sign_p": p, "perm_p": perm_p(d)}
    return out


def worst_metric() -> dict:
    """Per seed: each arm's value / best arm's value on HV, feasible, new high-margin and best PAE; score = minimum ratio."""
    ps = per_seed_table().set_index(["arm", "seed"])
    cols = {"hv2": "hv", "feasible": "feasible", "new_high_margin": "k15_new", "best_pae": "best_pae"}
    vals = {k: pd.DataFrame({a: [float(ps.loc[(a, s), c]) for s in SEEDS] for a in ARMS}, index=SEEDS) for k, c in cols.items()}
    if any((v.max(axis=1) <= 0).any() for v in vals.values()):
        raise SystemExit("a metric has no positive value on some seed; ratio undefined")
    ratios = {k: v.div(v.max(axis=1), axis=0) for k, v in vals.items()}
    score = pd.concat(ratios.values()).groupby(level=0).min().loc[SEEDS]
    within = {f"{x:g}": {a: int(np.all([ratios[k].loc[SEEDS, a].to_numpy() >= 1 - x - 1e-12 for k in ratios], axis=0).sum()) for a in ARMS} for x in X_GRID}
    tests = {}
    for a in ARMS:
        if a == MAIN:
            continue
        d = score[MAIN].to_numpy() - score[a].to_numpy()
        nz = d[d != 0]
        tests[a] = {"diff": float(d.mean()), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p": float(wilcoxon(nz).pvalue) if len(nz) else 1.0}
    weakest = {a: {k: int(sum(ratios[k].loc[s, a] <= score.loc[s, a] + 1e-12 for s in SEEDS)) for k in ratios} for a in ARMS}
    return {"per_seed": {a: [float(v) for v in score[a]] for a in ARMS}, "mean": {a: float(score[a].mean()) for a in ARMS},
            "min": {a: float(score[a].min()) for a in ARMS}, "within_all_four": within, "tests_vs_main": tests, "weakest_metric_counts": weakest}


def phase_rates() -> dict:
    ids, cont = DESIGN_COLS[:2], DESIGN_COLS[2:]
    def _aug(a: str, s: int) -> pd.DataFrame:
        return H[a, s].loc[H[a, s].phase == "aug"].reset_index(drop=True)
    def _keys(d: pd.DataFrame) -> set[str]:  # order-insensitive: one seed simulated the same 15 designs in a different order
        return set(d[ids].astype(str).agg("|".join, axis=1) + "|" + d[cont].round(6).astype(str).agg("|".join, axis=1))
    same = all(_keys(_aug(MAIN, s)) == _keys(_aug("fa15q", s)) for s in SEEDS)
    aug = pd.concat([H[MAIN, s][H[MAIN, s].phase == "aug"] for s in SEEDS], ignore_index=True)
    bo = pd.concat([H[a, s][H[a, s].phase == "opt"] for a in ARMS for s in SEEDS], ignore_index=True)
    a_hm, a_n, b_hm, b_n = int(margin_rich(aug).sum()), len(aug), int(margin_rich(bo).sum()), len(bo)
    pi_lo, pi_hi = beta.ppf(0.025, a_hm, b_hm + 1), beta.ppf(0.975, a_hm + 1, b_hm)
    to_ratio = lambda pi: pi / (1 - pi) * b_n / a_n  # noqa: E731
    bo_by_arm = {a: int(sum(margin_rich(H[a, s][H[a, s].phase == "opt"]).sum() for s in SEEDS)) for a in ARMS}
    return {"aug_batches_identical_in_fa15q": bool(same), "aug": {"high_margin": a_hm, "sims": a_n, "rate": a_hm / a_n},
            "bo_all_arms": {"high_margin": b_hm, "sims": b_n, "rate": b_hm / b_n, "by_arm": bo_by_arm},
            "rate_ratio": (a_hm / a_n) / (b_hm / b_n), "rate_ratio_ci95": [float(to_ratio(pi_lo)), float(to_ratio(pi_hi))]}


def feasible_rate() -> dict:
    lo, hi = WINDOW
    rate, n_rows = {}, set()
    for a in ARMS:
        r = []
        for s in SEEDS:
            h = H[a, s]; w = h["sim_index"].between(lo, hi)
            n_rows.add(int(w.sum())); r.append(float(feasible_mask(h)[w].sum()) / int(w.sum()))
        rate[a] = np.array(r)
    fa, qn = rate[MAIN], rate["qn"]
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(SEEDS), size=(N_BOOT, len(SEEDS)))
    ratio_boot = fa[idx].mean(axis=1) / qn[idx].mean(axis=1)
    cuts = np.arange(lo, hi + 1)
    lead = np.array([np.mean([feasible_mask(H[MAIN, s])[H[MAIN, s]["sim_index"] <= c].sum() - feasible_mask(H["qn", s])[H["qn", s]["sim_index"] <= c].sum() for s in SEEDS]) for c in cuts])
    slope, icpt = np.polyfit(cuts, lead, 1)
    r2 = 1 - np.sum((lead - (slope * cuts + icpt)) ** 2) / np.sum((lead - lead.mean()) ** 2)
    tests = {a: {"diff": float((fa - rate[a]).mean()), "p": float(wilcoxon((fa - rate[a])[(fa - rate[a]) != 0]).pvalue)} for a in ARMS if a != MAIN}
    return {"window": list(WINDOW), "rows_in_window": sorted(n_rows), "mean_rate": {a: float(v.mean()) for a, v in rate.items()},
            "ratio_fa_qn": float(fa.mean() / qn.mean()), "ratio_ci95": [float(np.percentile(ratio_boot, 2.5)), float(np.percentile(ratio_boot, 97.5))],
            "lead_vs_qn": {"slope_per_sim": float(slope), "r2": float(r2), "at_116": float(lead[0]), "at_181": float(lead[-1])}, "tests_vs_main": tests}


def main() -> None:
    res = {"definitions": {"joint_target": "first sim_index with cumulative strictly feasible >= F and cumulative new (sim_index > 100) high-margin >= M",
                           "time_tests": "unreached runs counted at 182; saved = other minus FA-MOBO; bootstrap CI (10,000, default_rng(0)); exact sign test (unreached = later); exact paired sign-flip permutation p",
                           "worst_metric": "min over {2-D HV ref (0,0), strictly feasible, new high-margin, best feasible PAE} of value / best of the five arms on the seed; Wilcoxon on non-zero differences",
                           "phase_rates": "high-margin designs per simulation: 150 augmentation simulations (FA-MOBO; identical in Aug-BO) vs every BO-phase simulation of the five arms; Clopper-Pearson conditional CI of the rate ratio",
                           "feasible_rate": "strictly feasible designs per simulation over simulations 116-181; paired bootstrap of the ratio of means"},
           "sensitivity": {f"M{m}_F{f}": time_block(f, m) for m in M_GRID for f in F_GRID},
           "worst_metric": worst_metric(), "phase_rates": phase_rates(), "feasible_rate": feasible_rate()}
    (OUT / "novel_numbers.json").write_text(json.dumps(res, indent=1))
    print("wrote novel_numbers.json")


if __name__ == "__main__":
    main()
