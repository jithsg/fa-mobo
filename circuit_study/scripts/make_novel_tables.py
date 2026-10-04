"""Supplementary tables for the reliability additions, from out/novel_numbers.json (written by make_novel.py)."""
from __future__ import annotations

import json

from common import ARMS, DATA, OUT

NUM = json.loads((OUT / "novel_numbers.json").read_text())
ORDER = ["fa15qc", "fa15q", "qnc", "qn", "qnm"]
SHORT = {a: ARMS[a][0] for a in ORDER}
F_SHOWN = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]


def _p(p: float) -> str:
    return "0.002" if abs(p - 0.001953125) < 1e-9 else f"{p:.3f}"


def _cmp(v: dict) -> str:
    return f"{v['saved_mean']:.1f} & {v['sooner']}/{v['later']} & {_p(v['sign_p'])} & {_p(v['perm_p'])}"


def table_sensitivity() -> str:
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Sensitivity of the joint target to its thresholds. A run reaches the target at the first simulation by which it has at least $F$ strictly feasible designs and at least $M$ new high-margin designs. Runs: runs of ten that reach it within 181 simulations (FA-MOBO / Aug-BO / Clf-BO). Saved: mean over seeds of the other arm's simulations to the target minus FA-MOBO's, with unreached runs counted at 182. S/L: seeds on which FA-MOBO is sooner / later. $p_{\mathrm{sign}}$: exact sign test (unreached counts as later). $p_{\mathrm{perm}}$: exact paired sign-flip permutation test of the mean saving. The main text uses $F=10$, $M=1$ (bold).}",
             r"\label{tab:s_sensitivity}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{rr c rrrr c rrrr}", r"\hline",
             r" & & & \multicolumn{4}{c}{against Aug-BO} & & \multicolumn{4}{c}{against Clf-BO} \\",
             r"$M$ & $F$ & Runs & Saved & S/L & $p_{\mathrm{sign}}$ & $p_{\mathrm{perm}}$ & & Saved & S/L & $p_{\mathrm{sign}}$ & $p_{\mathrm{perm}}$ \\", r"\hline"]
    for m in (1, 2):
        for f in F_SHOWN:
            b = NUM["sensitivity"][f"M{m}_F{f}"]
            runs = "/".join(str(b["reached"][a]) for a in ("fa15qc", "fa15q", "qnc"))
            row = f"{m} & {f} & {runs} & {_cmp(b['vs']['fa15q'])} & & {_cmp(b['vs']['qnc'])} \\\\"
            if (m, f) == (1, 10):
                row = " & ".join(rf"\textbf{{{c.strip()}}}" if c.strip() else c for c in row[:-2].split("&")) + r" \\"
            lines.append(row)
        if m == 1:
            lines.append(r"\hline")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def table_worst_metric() -> str:
    w = NUM["worst_metric"]
    labels = {"hv2": "hypervolume", "feasible": "feasible", "new_high_margin": "new high-margin", "best_pae": "best $PAE$"}
    head = " & ".join(SHORT[a] for a in ORDER)
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Worst-metric relative score at 181 simulations. On each seed, every arm's two-dimensional hypervolume, strictly feasible count, new high-margin count and best feasible $PAE$ are divided by the best value of the five arms on that seed; a run's score is the smallest of the four ratios (1 = best or tied-best on every metric). Within $X$\%: runs whose four ratios are all at least $1-X/100$. Weakest metric: the metric that sets the score, counted over the ten seeds (ties counted for each tied metric). $p$: Wilcoxon signed-rank test of FA-MOBO minus the arm on the non-zero per-seed differences.}",
             r"\label{tab:s_worst_metric}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l" + "r" * len(ORDER) + "}", r"\hline", rf" & {head} \\", r"\hline",
             "Mean score & " + " & ".join(f"{w['mean'][a]:.2f}" for a in ORDER) + r" \\",
             "Lowest run & " + " & ".join(f"{w['min'][a]:.2f}" for a in ORDER) + r" \\"]
    for x in ("0.05", "0.1", "0.25"):
        lines.append(f"Within {float(x) * 100:g}\\% on all four (runs) & " + " & ".join(str(w["within_all_four"][x][a]) for a in ORDER) + r" \\")
    lines.append(r"\hline")
    for k, lab in labels.items():
        lines.append(f"Weakest metric: {lab} & " + " & ".join(str(w["weakest_metric_counts"][a][k]) for a in ORDER) + r" \\")
    lines.append(r"\hline")
    t = w["tests_vs_main"]
    lines.append(r"FA-MOBO higher/lower (seeds) & -- & " + " & ".join(f"{t[a]['wins']}/{t[a]['losses']}" for a in ORDER[1:]) + r" \\")
    lines.append(r"$p$ against FA-MOBO & -- & " + " & ".join(_p(t[a]["p"]) for a in ORDER[1:]) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


CLF_CSV = DATA / "precomputed" / "T3c_2_first_rf_tests.csv"
CLF_ROWS = [  # (statistic in T3c_2_first_rf_tests.csv, label, decimals)
    ("n_pos", "Feasible training examples", 1),
    ("p_bound_feas_oos", r"$p_{\mathrm{RF}}$, feasible designs at the width bound", 2),
    ("p_lowk_oos", r"$p_{\mathrm{RF}}$, feasible designs with $K<10$", 2),
    ("p_lm_oos", r"$p_{\mathrm{RF}}$, feasible designs with $K\geq15$", 2),
    ("auc_oos", "AUC for strict feasibility", 3),
    ("imp_Ldc3", r"Importance, $L_{\mathrm{dc3}}$", 3),
    ("imp_Lx", r"Importance, $L_x$", 3),
    ("imp_width", r"Importance, $W_{\mathrm{M3}}$", 3),
    ("imp_Cm", r"Importance, $C_m$", 3),
    ("imp_CB", r"Importance, $C_B$", 3),
    ("imp_Cs", r"Importance, $C_s$", 3),
]


def table_classifier() -> str:
    import pandas as pd
    t = pd.read_csv(CLF_CSV).set_index("stat")
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Random-forest feasibility classifier before FA-MOBO's first BO batch, trained on the shared seed alone (seed only, 100 designs) or on the seed plus the augmentation batch (augmented, 115 designs), with the in-loop settings (300 trees, minimum leaf size 2, balanced-subsample class weights). $p_{\mathrm{RF}}$ and AUC are evaluated on the distinct converged designs of the five arms that lie outside the classifier's training set; the width bound is $W_{\mathrm{M3}}\leq641\,\mu$m. Importance: impurity-based feature importance, summed over the three embedding coordinates of each inductor. Values are means over ten seeds; difference: augmented minus seed-only, with a bootstrap 95\% interval (10\,000 resamples); higher: seeds on which the augmented value is higher; $p$: Wilcoxon signed-rank test over seeds.}",
             r"\label{tab:s_classifier}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrcr}", r"\hline",
             r"Statistic & Seed only & Augmented & Difference [95\% interval] & Higher & $p$ \\", r"\hline"]
    for stat, lab, d in CLF_ROWS:
        r = t.loc[stat]
        lines.append(f"{lab} & {r.lhs:.{d}f} & {r.aug:.{d}f} & {r['diff']:+.{d}f} [{r.ci_lo:+.{d}f}, {r.ci_hi:+.{d}f}] & {int(r.n_pos)}/10 & {_p(float(r.p_wilcoxon))} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    (OUT / "tab_s_sensitivity.tex").write_text(table_sensitivity())
    (OUT / "tab_s_worst_metric.tex").write_text(table_worst_metric())
    (OUT / "tab_s_classifier.tex").write_text(table_classifier())
    print("wrote tab_s_sensitivity.tex, tab_s_worst_metric.tex, tab_s_classifier.tex")


if __name__ == "__main__":
    main()
