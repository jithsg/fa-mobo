"""LaTeX tables and numbers.json for the revised manuscript, generated from the campaign data.

Run from the scripts folder:  python make_tables.py [--out DIR]

Tables: tab_ms_comparison (outcomes per arm), tab_ms_tests (paired tests against FA-MOBO),
tab_phase_origin, tab_hv_progression, tab_corners. tab_cost is no longer generated; the run-time
numbers stay in numbers.json. Metric definitions follow framing/FINAL_SPEC_FA-MOBO.md section 6.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from common import (ARMS, BUDGET, DESIGN_VARS, EA_METHODS, K_MARGIN, N_SEED, OP1DB_MARGIN, OUT, ROB, SEEDS, ea_outcomes, load_ea, load_hist,
                    margin_rich, per_seed_table, run_minutes)

CONT_VARS = [c for c in DESIGN_VARS if c not in ("Lx_id", "Ldc3_id")]
from problem import feasible_mask  # noqa: E402

from common import DATA  # noqa: E402
from T3a_metrics import K_CAP, K_REF, PSAT_REF, feasible_set, hv3  # noqa: E402

PS = per_seed_table()
NUM: dict = {}   # every number quoted in the text, for verification

MAIN = "fa15qc"
OTHERS = [a for a in ARMS if a != MAIN]
N_BOOT, BOOT_SEED = 10_000, 0           # bootstrap of the mean paired difference
F_TARGET, M_TARGET = 10, 1              # joint target: >= 10 strictly feasible and >= 1 new high-margin
QNM_MARGIN_DB = 0.3                     # OP1dB margin of the qNEHVI+m arm (run protocol)
T3A_CSV = DATA / "precomputed" / "T3a_per_run.csv"
HV3_TOL = 1e-9
# per-run metric -> (label in the tests table, decimals of the difference)
T3_METRICS = {
    "feasible": ("Feasible", 1),
    "high_margin": ("High-margin designs", 1),
    "hv3": ("Margin-aware HV", 0),
    "hv2": ("2-D HV", 0),
    "best_pae": ("Best PAE (\\%)", 1),
    "best_pae_hm": ("Best PAE, high-margin (\\%)$^\\dagger$", 1),
}


def ms(x, d=1):
    return f"{np.mean(x):.{d}f} $\\pm$ {np.std(x, ddof=1):.{d}f}"


def pval(a: str, b: str, met: str) -> tuple[float, int, float]:
    x = (PS[PS.arm == a].set_index("seed")[met] - PS[PS.arm == b].set_index("seed")[met]).loc[SEEDS]
    nz = x[x != 0]
    p = wilcoxon(nz).pvalue if len(nz) else 1.0
    return float(x.mean()), int((x > 0).sum()), float(p)


def fmt_p(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def _signed(v: float, d: int) -> str:
    s = f"{v:+.{d}f}"
    if float(s) == 0:
        return f"{0:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


# ----------------------------------------------------------------------------- legacy per-arm numbers (kept keys)
def core_numbers() -> None:
    """Per-arm blocks and vs_<arm> blocks used by make_figures.py and the manuscript (keys unchanged)."""
    for arm in ARMS:
        d = PS[PS.arm == arm]
        if arm != MAIN:
            ph, wh, p1 = pval(MAIN, arm, "hv"); pf, wf, p2 = pval(MAIN, arm, "feasible"); pk, wk, p3 = pval(MAIN, arm, "k15")
            NUM[f"vs_{arm}"] = {"hv_diff": ph, "hv_wins": wh, "hv_p": p1, "feas_diff": pf, "feas_wins": wf, "feas_p": p2, "k15_diff": pk, "k15_wins": wk, "k15_p": p3}
        NUM[arm] = {"hv_mean": float(d.hv.mean()), "hv_sd": float(d.hv.std(ddof=1)), "feasible_mean": float(d.feasible.mean()), "feasible_sd": float(d.feasible.std(ddof=1)),
                    "k15_mean": float(d.k15.mean()), "k15_sd": float(d.k15.std(ddof=1)), "best_pae_mean": float(d.best_pae.mean()), "k15_min": int(d.k15.min()), "k15_max": int(d.k15.max()),
                    "feasible_min": int(d.feasible.min()), "feasible_max": int(d.feasible.max()), "hv_min": float(d.hv.min()), "hv_max": float(d.hv.max())}


def cost_numbers() -> None:
    """Wall-clock minutes per run (queue-confounded); kept in numbers.json, no longer tabulated."""
    rm = run_minutes(); rm = rm[rm.arm.isin(ARMS)]
    for arm in ARMS:
        v = rm[rm.arm == arm].minutes
        NUM[arm].update({"minutes_median": float(v.median()), "minutes_min": int(v.min()), "minutes_max": int(v.max()), "runs_timed": int(len(v))})


# ----------------------------------------------------------------------------- T3 per-run metrics
def _first_index(cond: np.ndarray, sim_index: np.ndarray) -> float:
    """sim_index of the first row where `cond` holds; NaN when it never does."""
    return float(sim_index[np.argmax(cond)]) if cond.any() else np.nan


def _reach_indices(arm: str, seed: int) -> dict[str, float]:
    """First simulation index at which the joint target and its two parts are met (NaN = not by 181)."""
    h = load_hist(arm, seed)
    idx = h["sim_index"].to_numpy()
    cum_f = np.cumsum(feasible_mask(h).to_numpy())
    cum_m = np.cumsum((margin_rich(h) & (h["sim_index"] > N_SEED)).to_numpy())
    return {"reach_joint": _first_index((cum_f >= F_TARGET) & (cum_m >= M_TARGET), idx),
            "reach_f10": _first_index(cum_f >= F_TARGET, idx), "reach_m1": _first_index(cum_m >= M_TARGET, idx)}


def _hv3_run(arm: str, seed: int) -> float:
    """Margin-aware hypervolume in (PAE %, Psat dBm, min(K, K_CAP)), reference (0, PSAT_REF, K_REF)."""
    f = feasible_set(arm, seed, BUDGET)
    return hv3(np.c_[f["PAE_percent"], f["Psat_dBm"], f["Kc"]], (0.0, PSAT_REF, K_REF))


def t3_per_run() -> pd.DataFrame:
    """One row per (arm, seed) with every metric of the main comparison."""
    rows = []
    for _, r in PS.iterrows():
        rows.append({"arm": r.arm, "seed": int(r.seed), "feasible": r.feasible, "high_margin": r.k15, "high_margin_new": r.k15_new,
                     "hv3": _hv3_run(r.arm, int(r.seed)), "hv2": r.hv, "best_pae": r.best_pae, "best_pae_hm": r.best_k15_pae,
                     **_reach_indices(r.arm, int(r.seed))})
    return pd.DataFrame(rows)


def verify_hv3(t3: pd.DataFrame) -> float:
    """Reproduce column P1_hv3_K of T3a_per_run.csv (budget 181); stop if any run differs."""
    ref = pd.read_csv(T3A_CSV)
    ref = ref[ref.budget == BUDGET][["arm", "seed", "P1_hv3_K"]]
    m = t3.merge(ref, on=["arm", "seed"], how="left")
    if m["P1_hv3_K"].isna().any():
        raise SystemExit(f"hv3 check: {int(m['P1_hv3_K'].isna().sum())} runs missing from {T3A_CSV}")
    err = float((m["hv3"] - m["P1_hv3_K"]).abs().max())
    if err > HV3_TOL:
        raise SystemExit(f"hv3 check FAILED: max |diff| = {err:.3g} against {T3A_CSV}")
    print(f"hv3 check: reproduces P1_hv3_K for {len(m)} runs (max |diff| {err:.2g})")
    return err


# ----------------------------------------------------------------------------- statistics
def _col(t3: pd.DataFrame, arm: str, met: str) -> np.ndarray:
    return t3[t3.arm == arm].set_index("seed").loc[SEEDS, met].to_numpy(dtype=float)


def _summary(x: np.ndarray) -> dict:
    return {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)), "min": float(np.min(x)), "max": float(np.max(x)), "per_seed": [float(v) for v in x]}


def _median_reach(t: np.ndarray) -> dict:
    hit = t[~np.isnan(t)]
    return {"median": float(np.median(hit)) if len(hit) else None, "runs": int(len(hit)), "per_seed": [None if np.isnan(v) else int(v) for v in t]}


def paired_test(fa: np.ndarray, other: np.ndarray) -> dict:
    """FA-MOBO minus arm per seed: mean, bootstrap 95% CI, wins/ties/losses, Wilcoxon on non-zero differences."""
    d = fa - other
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(d), size=(N_BOOT, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()), "p": p}


def mcnemar_exact(x: np.ndarray, y: np.ndarray) -> dict:
    """Exact McNemar on paired booleans (reached / not reached)."""
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    return {"fa_only": b, "arm_only": c, "p": 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)}


def sign_test_sooner(ta: np.ndarray, tb: np.ndarray) -> dict:
    """Exact sign test: seeds on which FA-MOBO reaches the target sooner (unreached = later; equal = tie)."""
    a, b = np.nan_to_num(ta, nan=np.inf), np.nan_to_num(tb, nan=np.inf)
    k, later = int((a < b).sum()), int((a > b).sum())
    return {"sooner": k, "ties": int(len(a) - k - later), "later": later, "p": 1.0 if k + later == 0 else float(binomtest(k, k + later, 0.5).pvalue)}


def factorial_effects(t3: pd.DataFrame) -> dict:
    """2x2 factorial (augmentation x classifier): per-seed main effects and interaction, Wilcoxon on non-zero values."""
    out = {}
    for met in ("feasible", "high_margin"):
        fa, aug, clf, qn = (_col(t3, a, met) for a in ("fa15qc", "fa15q", "qnc", "qn"))
        effects = {"classifier": ((fa - aug) + (clf - qn)) / 2, "augmentation": ((fa - clf) + (aug - qn)) / 2, "interaction": fa - aug - clf + qn}
        out[met] = {k: {"mean": float(v.mean()), "p": float(wilcoxon(v[v != 0]).pvalue) if (v != 0).any() else 1.0} for k, v in effects.items()}
    return out


def t3_numbers(t3: pd.DataFrame) -> dict:
    per_arm, vs = {}, {}
    for arm in ARMS:
        blk = {met: _summary(_col(t3, arm, met)) for met in [*T3_METRICS, "high_margin_new"]}
        rj = _median_reach(_col(t3, arm, "reach_joint"))
        blk["joint_target"] = {"runs_reached": rj["runs"], "median_reach": rj["median"], "reach_per_seed": rj["per_seed"]}
        blk["to_10_feasible"] = _median_reach(_col(t3, arm, "reach_f10"))
        blk["to_first_new_high_margin"] = _median_reach(_col(t3, arm, "reach_m1"))
        per_arm[arm] = blk
    for arm in OTHERS:
        v = {met: paired_test(_col(t3, MAIN, met), _col(t3, arm, met)) for met in T3_METRICS}
        ta, tb = _col(t3, MAIN, "reach_joint"), _col(t3, arm, "reach_joint")
        v["joint_target_mcnemar"] = mcnemar_exact(~np.isnan(ta), ~np.isnan(tb))
        v["joint_target_sooner"] = sign_test_sooner(ta, tb)
        vs[arm] = v
    return {"definitions": {"strictly_feasible": "problem.feasible_mask over the first 181 rows", "high_margin": f"strictly feasible and OP1dB >= {OP1DB_MARGIN} dBm and nominal minKf >= {K_MARGIN:g}; includes the shared seed's designs",
                            "new": f"sim_index > {N_SEED}", "joint_target": f"first sim_index with cumulative strictly feasible >= {F_TARGET} and cumulative new high-margin >= {M_TARGET}",
                            "hv3": f"hv3 (T3a_metrics.py) of distinct strictly feasible designs in (PAE %, Psat dBm, min(K, {K_CAP:g})), reference (0, {PSAT_REF:g}, {K_REF:g})",
                            "hv2": "dominated hypervolume in (PAE %, Psat dBm), reference (0, 0)", "best_pae_hm": "post hoc: max PAE over high-margin designs",
                            "paired": f"FA-MOBO minus arm per seed; bootstrap {N_BOOT} resamples, numpy default_rng({BOOT_SEED}) per comparison; Wilcoxon signed-rank on non-zero differences",
                            "joint_target_tests": "exact McNemar (reached / not reached); exact sign test on sooner vs later (unreached = later, equal = tie)"},
            "arm_order": list(ARMS), "per_arm": per_arm, "vs": vs, "factorial": factorial_effects(t3)}


# ----------------------------------------------------------------------------- tables
def _caption_main() -> str:
    return (r"\caption{Outcomes after 181 harmonic-balance simulations per run (100 shared LHS designs + 81 method-specific): the five Bayesian arms, mean $\pm$ s.d. over ten paired seeds, and four evolutionary optimizers, one run each. "
            r"Arms: FA-MOBO, 15-design feasibility augmentation followed by 66 simulations of classifier-guided constrained qLogNEHVI (proposed); Clf-BO, classifier-guided BO without augmentation; "
            r"Aug-BO, augmentation followed by constrained qNEHVI without the classifier; qNEHVI+m, constrained qNEHVI with a " + f"{QNM_MARGIN_DB:g}" + r"\,dB $OP_{1dB}$ margin; qNEHVI, constrained qNEHVI. "
            r"Feasible: strictly feasible designs, meeting all six constraints. "
            r"High-margin: strictly feasible with $OP_{1dB}\geq" + f"{OP1DB_MARGIN:g}" + r"$\,dBm and nominal Rollett $K\geq" + f"{K_MARGIN:g}" + r"$, including the two such designs of the shared seed. "
            r"Joint target: at least " + f"{F_TARGET}" + r" strictly feasible designs and at least " + f"{M_TARGET}" + r" new high-margin design (simulation index $>" + f"{N_SEED}" + r"$); "
            r"runs: runs that meet it within 181 simulations; median sim.: median simulation index at which it is first met, over those runs (in parentheses when a single run meets it). "
            r"Margin-aware HV: dominated hypervolume of the strictly feasible designs in ($PAE$\,\%, $P_{\mathrm{sat}}$\,dBm, $K$ capped at " + f"{K_CAP:g}" + r"), reference point (0, " + f"{PSAT_REF:g}, {K_REF:g}" + r"). "
            r"2-D HV: dominated hypervolume in ($PAE$\,\%, $P_{\mathrm{sat}}$\,dBm), reference point (0, 0). Best PAE: highest $PAE$ among the strictly feasible designs of a run. "
            r"$^\dagger$Post hoc metric: highest $PAE$ among the high-margin designs of a run. "
            r"The evolutionary optimizers (NSGA-II, SMS-EMOA, GDE3 and the swarm method CMOPSO, run with pymoo) started from the same seed and are scored with the same definitions; their rows are single runs and are not part of the paired tests against FA-MOBO in Table~\ref{tab:ms_tests}.}")


def _fmt_median(v: float) -> str:
    """Median simulation index: integer as is, otherwise one decimal (e.g. 125.5)."""
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"


def _median_cell(blk: dict) -> str:
    j = blk["joint_target"]
    if not j["runs_reached"]:
        return "--"
    v = _fmt_median(j["median_reach"])
    return v if j["runs_reached"] > 1 else f"({v})"


def table_main(t3: pd.DataFrame) -> str:
    """Main comparison: outcomes per arm at 181 simulations, 10 seeds."""
    L = [r"\begin{table*}[pos=htbp]", r"\centering", _caption_main(), r"\label{tab:ms_comparison}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lcccccccc}", r"\toprule",
         r" & & & \multicolumn{2}{c}{Joint target} & & & & \\", r"\cmidrule(lr){4-5}",
         r"Method & Feasible & High-margin & Runs & \makecell{Median\\sim.} & \makecell{Margin-\\aware HV} & 2-D HV & \makecell{Best\\PAE (\%)} & \makecell{Best PAE,\\high-margin (\%)$^\dagger$} \\", r"\midrule"]
    for arm, (short, *_) in ARMS.items():
        c = lambda met, d: ms(_col(t3, arm, met), d)  # noqa: E731
        blk = NUM["t3"]["per_arm"][arm]
        L.append(f"{short} & {c('feasible', 1)} & {c('high_margin', 1)} & {blk['joint_target']['runs_reached']}/{len(SEEDS)} & {_median_cell(blk)} & "
                 f"{c('hv3', 0)} & {c('hv2', 0)} & {c('best_pae', 1)} & {c('best_pae_hm', 1)} \\\\")
    L += [r"\midrule", r"\multicolumn{9}{@{}l}{\textit{Evolutionary optimizers, one run each from the same seed}} \\"]
    NUM["ea_rows"] = {}
    for m in EA_METHODS:
        e = ea_outcomes(m); NUM["ea_rows"][m] = e
        reached = e["joint_target"] is not None
        opt = lambda v, d: "--" if v is None else f"{v:.{d}f}"  # noqa: E731
        L.append(f"{m} & {e['feasible']} & {e['high_margin']} & {int(reached)}/1 & {'(' + format(e['joint_target'], '.0f') + ')' if reached else '--'} & "
                 f"{e['hv3']:.0f} & {e['hv2']:.0f} & {opt(e['best_pae'], 1)} & {opt(e['best_pae_hm'], 1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def _test_cell(t: dict, d: int) -> str:
    return (f"\\makecell{{{_signed(t['diff'], d)} [{_signed(t['ci95'][0], d)}, {_signed(t['ci95'][1], d)}]\\\\"
            f"{t['wins']}/{t['ties']}/{t['losses']}; {fmt_p(t['p'])}}}")


def table_tests() -> str:
    """Paired comparison of FA-MOBO against each arm, one row per outcome."""
    vs = NUM["t3"]["vs"]
    L = [r"\begin{table*}[pos=htbp]", r"\centering",
         r"\caption{Paired comparison of FA-MOBO against each arm over the ten seeds (outcomes and definitions as in Table~\ref{tab:ms_comparison}). "
         r"Continuous outcomes, first line: mean difference FA-MOBO minus the arm, with a bootstrap 95\% confidence interval (" + f"{N_BOOT:,}".replace(",", "{,}") + r" resamples); "
         r"second line: seeds on which FA-MOBO is higher / tied / lower, and the two-sided $p$-value of the Wilcoxon signed-rank test on the non-zero differences. "
         r"Joint target reached: seeds on which only FA-MOBO / only the arm meets the joint target within 181 simulations, exact McNemar test. "
         r"Joint target sooner: seeds on which FA-MOBO meets the joint target sooner / at the same simulation / later (a run that never meets it counts as later), exact sign test. "
         r"$p$-values are not adjusted for multiplicity. $^\dagger$Post hoc metric.}",
         r"\label{tab:ms_tests}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\cellset}{\renewcommand{\arraystretch}{1.0}}",
         r"\begin{tabular}{l" + "c" * len(OTHERS) + "}", r"\toprule",
         r"Outcome & " + " & ".join(f"vs {ARMS[a][0]}" for a in OTHERS) + r" \\", r"\midrule"]
    for met, (label, d) in T3_METRICS.items():
        L.append(label + " & " + " & ".join(_test_cell(vs[a][met], d) for a in OTHERS) + r" \\ \addlinespace[2pt]")
        if met == "high_margin":
            mc = [vs[a]["joint_target_mcnemar"] for a in OTHERS]; st = [vs[a]["joint_target_sooner"] for a in OTHERS]
            L.append(r"Joint target reached & " + " & ".join(f"{m['fa_only']}/{m['arm_only']}; {fmt_p(m['p'])}" for m in mc) + r" \\")
            L.append(r"Joint target sooner & " + " & ".join(f"{s['sooner']}/{s['ties']}/{s['later']}; {fmt_p(s['p'])}" for s in st) + r" \\ \addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def _unit_cont(d: pd.DataFrame, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (d[CONT_VARS].to_numpy(float) - lo) / (hi - lo)


def _seed_distance(h: pd.DataFrame) -> float:
    """Median distance of a run's new feasible designs to the nearest feasible seed design (continuous variables, seed-scaled)."""
    seed = h.iloc[:N_SEED]
    lo, hi = seed[CONT_VARS].to_numpy(float).min(0), seed[CONT_VARS].to_numpy(float).max(0)
    S = _unit_cont(seed[feasible_mask(seed)], lo, hi)
    new = h[feasible_mask(h) & (h["sim_index"] > N_SEED)]
    if not len(new) or not len(S):
        return float("nan")
    D = np.sqrt(((_unit_cont(new, lo, hi)[:, None, :] - S[None, :, :]) ** 2).sum(-1))
    return float(np.median(D.min(axis=1)))


def _repeats(h: pd.DataFrame) -> int:
    """Simulations spent on a design the run had already simulated."""
    k = h[DESIGN_VARS].round(6).astype(str).agg("|".join, axis=1)
    return int(len(h) - k.nunique())


def table_ea(t3: pd.DataFrame) -> str:
    """FA-MOBO against the evolutionary single runs at the same budget, with the seeds on which FA-MOBO beats every one of them."""
    eh = {m: load_ea(m) for m in EA_METHODS}
    fa_hist = {s: load_hist(MAIN, s) for s in SEEDS}
    e = NUM["ea_rows"]
    nan = float("nan")
    rows = [
        ("Best feasible $PAE$ (\\%)", _col(t3, MAIN, "best_pae"), {m: e[m]["best_pae"] for m in EA_METHODS}, "higher", 1),
        ("Hypervolume", _col(t3, MAIN, "hv2"), {m: e[m]["hv2"] for m in EA_METHODS}, "higher", 0),
        ("Margin-aware hypervolume", _col(t3, MAIN, "hv3"), {m: e[m]["hv3"] for m in EA_METHODS}, "higher", 0),
        ("Simulations to the joint target", _col(t3, MAIN, "reach_joint"),
         {m: (float(e[m]["joint_target"]) if e[m]["joint_target"] else nan) for m in EA_METHODS}, "lower", 0),
        ("Repeats", np.array([_repeats(fa_hist[s]) for s in SEEDS], float), {m: float(_repeats(eh[m])) for m in EA_METHODS}, "lower", 0),
        ("Distance to the seed", np.array([_seed_distance(fa_hist[s]) for s in SEEDS]),
         {m: _seed_distance(eh[m]) for m in EA_METHODS}, "higher", 2),
    ]
    L = [r"\begin{table*}[pos=htbp]", r"\centering",
         r"\caption{FA-MOBO against the evolutionary single runs at the same budget of 181 simulations. FA-MOBO: mean over the ten seeds with the range across seeds; the evolutionary columns are one run each, so these are not paired tests. "
         r"Ahead or level in: seeds on which FA-MOBO is at least as good as every evolutionary run of that row; the repeats row is the only tie, because GDE3 also repeats no simulation. For the joint target the table gives the mean over the ten seeds; the median quoted in the text is 125.5, and a dash means the run never met it. "
         r"Repeats: simulations spent on a design the run had already simulated. Distance to the seed: median distance of the feasible designs found after the seed to the nearest feasible design of the seed, over the four continuous variables scaled to the seed's range; a larger distance means the run left the neighbourhood of the seed designs. "
         r"Against the same runs FA-MOBO returns fewer high-margin designs (5.6 against up to 20) and a lower best $PAE$ among them (30.8\% against up to 34.4\%), as Table~\ref{tab:ms_comparison} shows.}",
         r"\label{tab:ea_comparison}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{l" + "c" * (len(EA_METHODS) + 2) + "}", r"\toprule",
         "At 181 simulations & FA-MOBO (10 runs) & " + " & ".join(EA_METHODS) + r" & Ahead or level in \\", r"\midrule"]
    out = {}
    for label, v, ea_vals, better, d in rows:
        vals = np.array([ea_vals[m] for m in EA_METHODS], float)
        best = np.nanmax(vals) if better == "higher" else np.nanmin(vals)
        wins = int((v >= best).sum() if better == "higher" else (v <= best).sum())   # ahead of, or level with, every evolutionary run
        cells = " & ".join("--" if np.isnan(ea_vals[m]) else f"{ea_vals[m]:.{d}f}" for m in EA_METHODS)
        L.append(f"{label} & {np.mean(v):.{d}f} ({np.min(v):.{d}f}--{np.max(v):.{d}f}) & {cells} & {wins}/{len(SEEDS)} \\\\")
        out[label] = {"fa_mean": float(np.mean(v)), "fa_min": float(np.min(v)), "fa_max": float(np.max(v)),
                      "ea": {m: (None if np.isnan(ea_vals[m]) else float(ea_vals[m])) for m in EA_METHODS},
                      "better_when": better, "fa_ahead_or_level_seeds": wins, "fa_per_seed": [float(x) for x in v]}
    NUM["ea_vs_fa"] = out
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table_phases() -> str:
    """Where feasible and high-margin designs come from: phase split, per run."""
    L = [r"\begin{table}[pos=htbp]", r"\centering", r"\caption{Origin of the feasible and high-margin designs, mean per run over ten seeds. The shared seed contributes 2 feasible designs (both high-margin) to every arm; augmentation is 15 designs (FA-MOBO, Aug-BO) and the BO phase 66 (81 for arms without augmentation).}",
         r"\label{tab:phase_origin}", r"\small", r"\begin{tabular}{lcccc}", r"\toprule", r"Method & Feasible (aug.) & Feasible (BO) & High-margin (aug.) & High-margin (BO) \\", r"\midrule"]
    for arm, (short, long, *_) in ARMS.items():
        d = PS[PS.arm == arm]
        na, no = int(d.n_aug.iloc[0]), int(d.n_opt.iloc[0])
        fa = f"{d.feasible_aug.mean():.1f} / {na}" if na else "--"
        ka = f"{d.k15_aug.mean():.1f} / {na}" if na else "--"
        L.append(f"{short} & {fa} & {d.feasible_opt.mean():.1f} / {no} & {ka} & {d.k15_opt.mean():.1f} / {no} \\\\")
        NUM[arm].update({"feasible_aug": float(d.feasible_aug.mean()), "feasible_opt": float(d.feasible_opt.mean()), "k15_aug": float(d.k15_aug.mean()), "k15_opt": float(d.k15_opt.mean()), "n_aug": na, "n_opt": no})
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table_progression() -> str:
    """HV and feasible count at the phase boundaries (100, 115, 181 sims)."""
    L = [r"\begin{table}[pos=htbp]", r"\centering", r"\caption{Dominated hypervolume and cumulative strictly feasible designs at the phase boundaries, mean $\pm$ s.d. over ten seeds. All arms share the 100-design seed (HV 503.9, 2 feasible).}",
         r"\label{tab:hv_progression}", r"\small", r"\begin{tabular}{lcccc}", r"\toprule", r"Method & HV @115 & HV @181 & Feasible @115 & Feasible @181 \\", r"\midrule"]
    for arm, (short, *_) in ARMS.items():
        h115 = [load_hist(arm, s).loc[N_SEED + 15 - 1, "cum_hv"] for s in SEEDS]; h181 = [load_hist(arm, s).loc[BUDGET - 1, "cum_hv"] for s in SEEDS]
        f115 = [load_hist(arm, s).loc[N_SEED + 15 - 1, "cum_feasible"] for s in SEEDS]; f181 = [load_hist(arm, s).loc[BUDGET - 1, "cum_feasible"] for s in SEEDS]
        L.append(f"{short} & {ms(h115, 0)} & {ms(h181, 0)} & {ms(f115)} & {ms(f181)} \\\\")
        NUM[arm].update({"hv_115": float(np.mean(h115)), "feasible_115": float(np.mean(f115))})
    h = load_hist("qn", 101); NUM["seed"] = {"hv_100": float(h.loc[N_SEED - 1, "cum_hv"]), "feasible_100": int(h.loc[N_SEED - 1, "cum_feasible"])}
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def _corner_label(c: str) -> str:
    return c.replace("C", "\\,$^\\circ$C").replace("/-", " $-$").replace("/", " ")


def corner_designs():
    """Corner-sweep rows for the paper's arms, with a design signature. A design selected in several seeds (chiefly the two
    shared-seed designs) was swept repeatedly with identical results, so every statistic counts each design once."""
    d = pd.read_csv(DATA / "corner_sweep" / "corner_sweep_fa15qc_qn.csv"); d = d[d.arm.isin(["fa15qc", "qn"])].copy()
    key = ["Lx_id", "Ldc3_id", "M3_width", "CB", "Cs", "Cm"]
    d["sig"] = d[key].round(6).astype(str).agg("|".join, axis=1); d["stable"] = d.minKf >= 1
    per_arm = d.drop_duplicates(["arm", "sig", "corner"])          # distinct designs within each arm
    pooled = d.drop_duplicates(["sig", "corner"])                  # distinct designs overall
    return d, per_arm, pooled


def _design_table(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = rows.groupby(by)
    return pd.DataFrame({"tt_K": g.tt_K.first(), "stable_all": g.minKf.min() >= 1, "specs_all": g.pass_strict.all()})


def table_corners() -> str:
    d, per_arm, pooled = corner_designs()
    corners = ["TT/27C", "FF/27C", "SS/27C", "FF/-40C", "SS/85C"]
    L = [r"\begin{table*}[pos=htbp]", r"\centering", r"\caption{Process--temperature corner sweep of the nominally margin-rich designs (strictly feasible, $OP_{1dB} \geq 15.8$\,dBm, $K \geq 10$ at TT/27\,$^\circ$C; up to the four highest-$PAE$ candidates per seed, all ten seeds; bias fixed). Each distinct design is counted once: designs selected in several seeds, chiefly the two shared-seed designs, were swept repeatedly with identical results. One design was selected by both arms, so the arm rows count 39 + 13 designs and the last row the 51 distinct designs. Stable: Rollett $K \geq 1$ at that corner; spec.: all six constraints met at that corner. Last two columns: designs stable at every corner, split by nominal $K$.}",
         r"\label{tab:corners}", r"\footnotesize", r"\setlength{\tabcolsep}{3.2pt}", r"\begin{tabular}{lc" + "cc" * len(corners) + r"cc}", r"\toprule",
         r"Designs & $n$ & " + " & ".join("\\multicolumn{2}{c}{" + _corner_label(c) + "}" for c in corners) + r" & \multicolumn{2}{c}{Stable at all 5 corners} \\",
         " ".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(corners))) + r" \cmidrule(lr){13-14}",
         r" & & " + " & ".join(["stab. & spec."] * len(corners)) + r" & $K\geq15$ & $10\leq K<15$ \\", r"\midrule"]
    groups = [("FA-MOBO", per_arm[per_arm.arm == "fa15qc"], "fa15qc"), ("qNEHVI", per_arm[per_arm.arm == "qn"], "qn"), ("All distinct", pooled, "all")]
    for name, rows, tag in groups:
        des = _design_table(rows, ["sig"]); cells = []
        for c in corners:
            sc = rows[rows.corner == c]; cells += [f"{sc.stable.mean() * 100:.0f}\\%", f"{sc.pass_strict.mean() * 100:.0f}\\%"]
        hi = des[des.tt_K >= K_MARGIN]; lo = des[des.tt_K < K_MARGIN]
        if tag == "all":
            L.append(r"\midrule")
        L.append(f"{name} & {len(des)} & " + " & ".join(cells) + f" & {int(hi.stable_all.sum())}/{len(hi)} & {int(lo.stable_all.sum())}/{len(lo)} \\\\")
        NUM[f"corner_{tag}"] = {"n": int(len(des)), "stable_all": int(des.stable_all.sum()), "specs_all_corners": int(des.specs_all.sum()),
                               "hi_stable": int(hi.stable_all.sum()), "hi_n": int(len(hi)), "lo_stable": int(lo.stable_all.sum()), "lo_n": int(len(lo)),
                               **{f"{c}_stable": float(rows[rows.corner == c].stable.mean()) for c in corners}, **{f"{c}_specs": float(rows[rows.corner == c].pass_strict.mean()) for c in corners},
                               "op1db_fail_ss85": float((rows[rows.corner == "SS/85C"].OP1dB_dBm < 15.5).mean())}
    des = _design_table(pooled, ["sig"]); hi = des[des.tt_K >= K_MARGIN]; lo = des[des.tt_K < K_MARGIN]
    from scipy.stats import fisher_exact
    NUM["corner_rule"] = {"hi_stable": int(hi.stable_all.sum()), "hi_n": int(len(hi)), "lo_stable": int(lo.stable_all.sum()), "lo_n": int(len(lo)),
                          "designs": int(len(des)), "selections": int(d.label.nunique()), "rows": int(len(d)),
                          "fisher_p": float(fisher_exact([[hi.stable_all.sum(), (~hi.stable_all).sum()], [lo.stable_all.sum(), (~lo.stable_all).sum()]]).pvalue),
                          "hi_lb95": float(0.025 ** (1.0 / len(hi))) if hi.stable_all.all() else None,
                          "k_gap": [float(lo.tt_K.max()), float(hi.tt_K.min())]}
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def extra_numbers() -> None:
    """Numbers quoted in the text that are not in a table."""
    _, per_arm, _ = corner_designs()
    tt = per_arm[per_arm.corner == "TT/27C"].set_index(["arm", "sig"]).PAE_percent; ss85 = per_arm[per_arm.corner == "SS/85C"].set_index(["arm", "sig"]).PAE_percent
    ret = (ss85 / tt)
    NUM["pae_retention_ss85"] = {a: float(ret.loc[a].median()) for a in ("fa15qc", "qn")}
    NUM["seeds"] = SEEDS; NUM["budget"] = BUDGET; NUM["n_seed"] = N_SEED


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    core_numbers()
    t3 = t3_per_run()
    NUM["t3"] = {**t3_numbers(t3), "hv3_check_max_abs_diff": verify_hv3(t3)}
    for name, fn in [("tab_ms_comparison", lambda: table_main(t3)), ("tab_ms_tests", table_tests), ("tab_phase_origin", table_phases),
                     ("tab_hv_progression", table_progression), ("tab_corners", table_corners), ("tab_ea_comparison", lambda: table_ea(t3))]:
        (out_dir / f"{name}.tex").write_text(fn() + "\n"); print("wrote", name)
    extra_numbers()
    (out_dir / "numbers.json").write_text(json.dumps(NUM, indent=1)); print("wrote numbers.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT, help="output directory for tab_*.tex and numbers.json")
    main(ap.parse_args().out)
