"""Emit the benchmark tables in the manuscript's house style from out/bench_numbers.json.

  python bench_tables.py                       # write tables into out/tables/
  python bench_tables.py --out <dir>           # write them somewhere else (e.g. the manuscript tables/ dir)

Three tables are produced: the main comparison, the paired tests against FA-MOBO, and the
target-fraction sweep. Evaluations to target is suppressed on OSY: a single feasible point
dominates that problem's hypervolume, so the threshold is crossed at each arm's first
model-driven evaluation and the apparent differences are phase structure, not search quality
(see the caveats in bench_report.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, norm

HERE = Path(__file__).resolve().parent
ARMS = ("FA-MOBO", "Aug-BO", "Clf-BO", "qNEHVI")
MAIN = "FA-MOBO"
NO_TARGET_METRIC = ("OSY",)     # hypervolume dominated by one point; see module docstring
DISPLAY = {"C2DTLZ2": "C2-DTLZ2", "C3DTLZ4": "C3-DTLZ4"}   # problem names as written in the text


def display_names(tex: str) -> str:
    for k, v in DISPLAY.items():
        tex = tex.replace(k, v)
    return tex


def _ptex(p: float) -> str:
    """p-value for a table cell: a p-value is never exactly zero, so one that rounds to 0.000
    is shown as an upper bound."""
    return "$p<0.001$" if p < 0.0005 else f"$p={p:.3f}$"


def load() -> dict:
    p = HERE / "out" / "bench_numbers.json"
    if not p.exists():
        raise SystemExit("run bench_report.py report first")
    return json.loads(p.read_text())


def _by_seed(rows: list[dict], prob: str, arm: str, key: str) -> dict[int, float]:
    return {r["seed"]: r[key] for r in rows if r["problem"] == prob and r["arm"] == arm and r[key] is not None}


def _ahead(rows: list[dict], prob: str, key: str, higher_is_better: bool,
           drop_seed_solved: bool = False) -> tuple[int, int]:
    """Seeds on which FA-MOBO is at least as good as every other arm, paired by seed.

    With drop_seed_solved, seeds whose target was already met during the seeding phase are
    excluded: that phase is shared by every arm, so such a seed is a tie by construction and
    counting it as "level" would credit the method for the initialization.
    """
    main = _by_seed(rows, prob, MAIN, key)
    others = {a: _by_seed(rows, prob, a, key) for a in ARMS if a != MAIN}
    seeds = [s for s in main if all(s in o for o in others.values())]
    if drop_seed_solved:
        solved = {r["seed"] for r in rows if r["problem"] == prob and r.get("target_reached_in_seed_phase")}
        seeds = [s for s in seeds if s not in solved]
    if not seeds:
        return 0, 0
    wins = 0
    for s in seeds:
        rest = [others[a][s] for a in others]
        wins += int(main[s] >= max(rest) if higher_is_better else main[s] <= min(rest))
    return wins, len(seeds)


def table_main(data: dict) -> str:
    """Per problem: hypervolume, feasible designs and evaluations to target for the four arms."""
    rows, per_arm = data["per_run"], data["per_arm"]
    metrics = (("Hypervolume", "hv_mean", "hv_sd", "hv", True, 3),
               ("Feasible designs", "feasible_mean", None, "feasible", True, 1),
               ("Evaluations to target", "median_evals_to_target", None, "evals_to_target", False, 0))
    L = [r"\begin{table*}[pos=htbp]", r"\centering",
         r"\caption{FA-MOBO against the three ablated arms on eight constrained multi-objective benchmarks, "
         r"20 seeds per arm and problem at a fixed evaluation budget. Every arm shares the seeding phase of a given seed, "
         r"so the arms are paired and differ only in what follows the seed. Hypervolume and feasible designs are the mean over seeds "
         r"(standard deviation in brackets for hypervolume); evaluations to target is the median over the seeds that reached half the "
         r"hypervolume of a 20,000-point random search, with the number of such seeds in brackets. "
         r"Ahead or level in: seeds on which FA-MOBO is at least as good as all three other arms. For evaluations to target, "
         r"seeds whose target was already met during the shared seeding phase are excluded from that count, because every arm receives the same "
         r"seed designs and such a seed is a tie by construction. "
         r"Evaluations to target is not reported for OSY, where one feasible point dominates the hypervolume, so the threshold is crossed "
         r"at each arm's first model-driven evaluation and the gaps reflect phase structure rather than search quality.}",
         r"\label{tab:bench_main}", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{ll" + "c" * len(ARMS) + "c}", r"\toprule",
         "Problem & Metric & " + " & ".join(ARMS) + r" & Ahead or level in \\", r"\midrule"]
    out: dict = {}
    probs = sorted(per_arm)
    for i, prob in enumerate(probs):
        for label, mkey, skey, rkey, higher, d in metrics:
            if label.startswith("Evaluations") and prob in NO_TARGET_METRIC:
                cells = " & ".join("--" for _ in ARMS)
                ahead = "--"
            else:
                parts = []
                for a in ARMS:
                    s = per_arm[prob].get(a)
                    if not s or s.get(mkey) is None:
                        parts.append("--"); continue
                    if skey:
                        sd_d = max(0, d - 2) if s[skey] >= 100 else d      # wide magnitudes (OSY) need no decimals
                        parts.append(f"{s[mkey]:.{max(0, d - 2) if s[mkey] >= 100 else d}f} ({s[skey]:.{sd_d}f})")
                    elif label.startswith("Evaluations"):
                        parts.append(f"{s[mkey]:.{d}f} ({s['runs_reaching_target']})")
                    else:
                        parts.append(f"{s[mkey]:.{d}f}")
                cells = " & ".join(parts)
                w, n = _ahead(rows, prob, rkey, higher, drop_seed_solved=label.startswith("Evaluations"))
                ahead = f"{w}/{n}" if n else "--"
            head = prob if label == metrics[0][0] else ""
            L.append(f"{head} & {label} & {cells} & {ahead} \\\\")
            out.setdefault(prob, {})[label] = {"cells": cells, "ahead": ahead}
        if i < len(probs) - 1:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    data.setdefault("tables", {})["main"] = out
    return "\n".join(L)


def table_tests(data: dict) -> str:
    """Paired Wilcoxon tests of FA-MOBO against each other arm, per problem."""
    tests = data["tests"]
    others = [a for a in ARMS if a != MAIN]
    L = [r"\begin{table}[H]", r"\centering",
         r"\caption{Paired comparison of FA-MOBO against each ablated arm on the benchmark problems. "
         r"Runs are paired by seed, because all arms share the seeding phase of a seed. Each cell gives the mean paired difference "
         r"(FA-MOBO minus the other arm), the seeds won and lost, and the two-sided Wilcoxon signed-rank $p$ on the non-zero differences. "
         r"A positive difference favours FA-MOBO for both metrics.}",
         r"\label{tab:s_bench_tests}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{ll" + "c" * len(others) + "}", r"\toprule",
         "Problem & Metric & " + " & ".join(f"vs {a}" for a in others) + r" \\", r"\midrule"]
    for prob in sorted(tests):
        for met, label, d in (("hv", "Hypervolume", 3), ("feasible", "Feasible designs", 1)):
            cells = []
            for a in others:
                t = tests[prob].get(met, {}).get(a)
                if not t:
                    cells.append("--"); continue
                cells.append(f"{t['diff_mean']:+.{d}f}, {t['wins']}/{t['losses']}, {_ptex(t['p'])}")
            head = prob if met == "hv" else ""
            L.append(f"{head} & {label} & " + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def table_sweep(data: dict) -> str:
    """Evaluations to target across target levels, with the share of runs the seed alone solved."""
    per_arm = data["per_arm"]
    fracs = sorted({k[len("reach_"):] for p in per_arm.values() for a in p.values() for k in a if k.startswith("reach_")},
                   key=float)
    L = [r"\begin{table}[H]", r"\centering",
         r"\caption{Evaluations to target at four target levels, as a fraction of the hypervolume reached by a 20,000-point random search. "
         r"Each cell is the median over the seeds that reached that level, with the number of such seeds in brackets; a dash means no seed reached it. "
         r"Seed: the share of runs whose target was already met during the seeding phase, which every arm shares, so those runs cannot separate the arms. "
         r"This is why the comparison is read from the whole sweep rather than from a single target level.}",
         r"\label{tab:s_bench_sweep}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{ll" + "c" * len(fracs) + "c}", r"\toprule",
         "Problem & Method & " + " & ".join(fracs) + r" & Seed \\", r"\midrule"]
    for prob in sorted(per_arm):
        if prob in NO_TARGET_METRIC:
            continue
        for a in ARMS:
            s = per_arm[prob].get(a)
            if not s:
                continue
            cells = []
            for f in fracs:
                e = s.get(f"reach_{f}")
                cells.append(f"{e['median_evals']:.0f} ({e['runs_reaching']})" if e and e["median_evals"] else "--")
            head = prob if a == ARMS[0] else ""
            L.append(f"{head} & {a} & " + " & ".join(cells) + f" & {s['share_target_in_seed_phase']:.0%}".replace("%", r"\%") + r" \\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def _common_seed_means(rows: list[dict], prob: str, metric: str) -> tuple[dict[str, float], dict[str, float], int]:
    """Mean and SD per arm over the seeds that every arm has run, so the comparison is paired."""
    per = {a: {r["seed"]: r[metric] for r in rows
               if r["problem"] == prob and r["arm"] == a and r[metric] is not None} for a in ARMS}
    seeds = sorted(set.intersection(*(set(v) for v in per.values()))) if all(per.values()) else []
    if not seeds:
        return {}, {}, 0
    mean = {a: float(np.mean([per[a][s] for s in seeds])) for a in ARMS}
    sd = {a: float(np.std([per[a][s] for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0 for a in ARMS}
    return mean, sd, len(seeds)


def _rank_matrix(rows: list[dict], per_arm: dict, metric: str) -> tuple[dict, dict, dict]:
    """Per-problem means, SDs and ranks (1 = best); higher is better for both reported metrics."""
    means, sds, ranks = {}, {}, {}
    for prob in sorted(per_arm):
        m, sd, n = _common_seed_means(rows, prob, metric)
        if not m:
            continue
        means[prob], sds[prob] = m, sd
        order = sorted(ARMS, key=lambda a: -m[a])
        ranks[prob] = {a: i + 1 for i, a in enumerate(order)}
    return means, sds, ranks


def table_ranks(data: dict, metric: str = "feasible", label: str = "feasible designs", dec: int = 2) -> str:
    """Mean (SD) per problem per arm with per-problem ranks, average rank and final rank."""
    rows, per_arm = data["per_run"], data["per_arm"]
    means, sds, ranks = _rank_matrix(rows, per_arm, metric)
    L = [r"\begin{table*}[pos=htbp]", r"\centering",
         r"\caption{Mean " + label + r" per benchmark problem for the four compared configurations, with the standard deviation in brackets "
         r"and the per-problem rank in italics (1 is best). Each problem uses only the seeds that all four configurations have run, so the "
         r"comparison is paired; the number of such seeds is given per row. Higher is better. The final two rows give the average rank over "
         r"the problems and the resulting overall ranking.}",
         r"\label{tab:bench_ranks_" + metric + "}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lc" + "c" * len(ARMS) + "}", r"\toprule",
         "Problem & Seeds & " + " & ".join(ARMS) + r" \\", r"\midrule"]
    for prob in sorted(means):
        _, _, n = _common_seed_means(rows, prob, metric)
        cells = " & ".join(f"{means[prob][a]:.{dec}f} ({sds[prob][a]:.{dec}f}) \\textit{{{ranks[prob][a]}}}" for a in ARMS)
        L.append(f"{prob} & {n} & {cells} \\\\")
    if ranks:
        avg = {a: float(np.mean([ranks[p][a] for p in ranks])) for a in ARMS}
        order = sorted(ARMS, key=lambda a: avg[a])
        final = {a: i + 1 for i, a in enumerate(order)}
        L.append(r"\midrule")
        L.append("Average rank & & " + " & ".join(f"{avg[a]:.2f}" for a in ARMS) + r" \\")
        L.append("Overall rank & & " + " & ".join(f"{final[a]}" for a in ARMS) + r" \\")
        data.setdefault("ranks", {})[metric] = {"avg_rank": avg, "overall": final,
                                                "per_problem": {p: ranks[p] for p in ranks}}
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def _holm(data: dict, metric: str) -> dict:
    """Friedman across problems, then Holm-Bonferroni post-hoc with FA-MOBO as control (Demsar's procedure)."""
    rows, per_arm = data["per_run"], data["per_arm"]
    means, _, ranks = _rank_matrix(rows, per_arm, metric)
    probs = sorted(ranks)
    k, N = len(ARMS), len(probs)
    if N < 3:
        return {"n_problems": N, "note": "too few problems for a Friedman test"}
    chi, pf = friedmanchisquare(*[[means[p][a] for p in probs] for a in ARMS])
    avg = {a: float(np.mean([ranks[p][a] for p in probs])) for a in ARMS}
    se = float(np.sqrt(k * (k + 1) / (6.0 * N)))
    comps = []
    for a in ARMS:
        if a == MAIN:
            continue
        z = (avg[a] - avg[MAIN]) / se          # positive z favours the control (lower rank is better)
        comps.append({"arm": a, "avg_rank": avg[a], "z": float(z), "p": float(2 * (1 - norm.cdf(abs(z))))})
    comps.sort(key=lambda c: c["p"])
    for i, c in enumerate(comps):              # Holm: compare the i-th smallest p against alpha/(k-1-i)
        c["alpha_holm"] = 0.05 / (k - 1 - i)
        c["reject"] = bool(c["p"] < c["alpha_holm"])
    return {"n_problems": N, "friedman_chi2": float(chi), "friedman_p": float(pf),
            "control": MAIN, "control_avg_rank": avg[MAIN], "comparisons": comps}


def table_holm(data: dict) -> str:
    """Friedman and Holm-Bonferroni results for both reported metrics."""
    L = [r"\begin{table}[H]", r"\centering",
         r"\caption{Friedman test across the benchmark problems followed by a Holm--Bonferroni post-hoc comparison with FA-MOBO as the "
         r"control configuration. Ranks are computed per problem from the paired means (1 is best), $z$ is the rank difference from the control "
         r"divided by its standard error, and $\alpha_{\mathrm{Holm}}$ is the step-down level for that position in the ordered sequence. "
         r"A positive $z$ favours FA-MOBO. With eight problems and four configurations the test has low power, so a non-rejection is not "
         r"evidence of equality.}",
         r"\label{tab:s_bench_holm}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llccccc}", r"\toprule",
         r"Metric & Configuration & Avg. rank & $z$ & $p$ & $\alpha_{\mathrm{Holm}}$ & Rejected \\", r"\midrule"]
    store = {}
    for metric, label in (("feasible", "Feasible designs"), ("hv", "Hypervolume")):
        h = _holm(data, metric)
        store[metric] = h
        if "comparisons" not in h:
            continue
        L.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{label}: Friedman $\\chi^2={h['friedman_chi2']:.2f}$, "
                 f"$p={h['friedman_p']:.3f}$, {h['n_problems']} problems}}}} \\\\")
        L.append(f" & {MAIN} (control) & {h['control_avg_rank']:.2f} & -- & -- & -- & -- \\\\")
        for c in h["comparisons"]:
            L.append(f" & {c['arm']} & {c['avg_rank']:.2f} & {c['z']:+.2f} & {c['p']:.3f} & {c['alpha_holm']:.4f} & "
                     f"{'yes' if c['reject'] else 'no'} \\\\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    data["holm"] = store
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "out" / "tables")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    data = load()
    for name, fn in (("tab_bench_main", table_main), ("tab_s_bench_tests", table_tests), ("tab_s_bench_sweep", table_sweep),
                     ("tab_bench_ranks_feasible", lambda d: table_ranks(d, "feasible", "feasible designs", 2)),
                     ("tab_bench_ranks_hv", lambda d: table_ranks(d, "hv", "hypervolume", 4)),
                     ("tab_s_bench_holm", table_holm)):
        (out / f"{name}.tex").write_text(display_names(fn(data)) + "\n")
        print("wrote", out / f"{name}.tex")


if __name__ == "__main__":
    main()
