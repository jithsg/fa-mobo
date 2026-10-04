"""Additional submission assets: median attainment fronts (figure), selected design across corners (table),
and optimizer settings (table read from the code). Run: python make_extra.py"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ARMS, DATA, EA_COLOR, EA_METHODS, FS, OUT, SEEDS, load_ea, load_hist, margin_rich, save, style  # noqa: E402
from problem import feasible_mask  # noqa: E402

style()
MAIN = "fa15qc"
KEY = ["Lx_id", "Ldc3_id", "M3_width", "CB", "Cs", "Cm"]


# ---------------------------------------------------------------- attainment fronts
def _run_front(h: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """g(x) = best Psat among strictly feasible designs with PAE >= x (−inf if none)."""
    f = h[feasible_mask(h)]
    pae, ps = f.PAE_percent.to_numpy(), f.Psat_dBm.to_numpy()
    return np.array([ps[pae >= x].max() if (pae >= x).any() else -np.inf for x in grid])


def _attainment(arm: str, grid: np.ndarray, k: int) -> np.ndarray:
    """k-th attainment surface over 10 runs: points (x, y) attained by at least k runs."""
    G = np.vstack([_run_front(load_hist(arm, s), grid) for s in SEEDS])
    return -np.sort(-G, axis=0)[k - 1]


def fig_attainment() -> None:
    grid = np.arange(20.0, 45.01, 0.02)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True)
    ax = axes[0]
    for arm, (short, _, col, ls, lw) in ARMS.items():
        y = _attainment(arm, grid, 5); ok = np.isfinite(y)
        ax.step(grid[ok], y[ok], where="post", color=col, ls=ls, lw=lw, label=short, zorder=4 if arm == MAIN else 3)
    ea = {m: _run_front(load_ea(m), grid) for m in EA_METHODS}
    drawn: list[str] = []
    for m, y in ea.items():
        if any(np.array_equal(ea[d], y) for d in drawn):
            continue
        same = [n for n in EA_METHODS if np.array_equal(ea[n], y)]
        ok = np.isfinite(y)
        solo = len(same) == 1 and drawn
        ax.step(grid[ok], y[ok], where="post", color=EA_COLOR, lw=0.9, ls="-" if not drawn else ":", zorder=2,
                marker="o" if solo else None, markevery=[int(np.flatnonzero(ok)[-1])] if solo else None, ms=4, mfc="white", mec=EA_COLOR, mew=0.8,
                label=", ".join(same) + (" (1 run each)" if len(same) > 1 else " (1 run)"))
        drawn.append(m)
    ax.set_title("(a) Median fronts (10 seeds); EA single runs"); ax.set_xlabel("PAE (%)"); ax.set_ylabel("Psat (dBm)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=FS["legend"], columnspacing=1.0, handletextpad=0.4)
    ax = axes[1]; col = ARMS[MAIN][2]
    for k, lab, alpha in ((1, "best run (1 of 10)", 0.35), (5, "median (5 of 10)", 1.0), (10, "worst run (10 of 10)", 0.6)):
        y = _attainment(MAIN, grid, k); ok = np.isfinite(y)
        ax.step(grid[ok], y[ok], where="post", color=col, lw=1.8 if k == 5 else 1.0, ls="-" if k == 5 else "--", alpha=alpha, label=f"FA-MOBO {lab}")
    yq = _attainment("qn", grid, 5); ok = np.isfinite(yq)
    ax.step(grid[ok], yq[ok], where="post", color=ARMS["qn"][2], lw=0.9, label="qNEHVI median")
    hm = pd.concat([load_hist(MAIN, s).pipe(lambda h: h[margin_rich(h)]) for s in SEEDS]).drop_duplicates(KEY)
    ax.scatter(hm.PAE_percent, hm.Psat_dBm, marker="D", s=16, facecolor=col, edgecolor="k", lw=0.4, zorder=5, label=f"FA-MOBO high-margin designs ({len(hm)})")
    ax.set_title("(b) FA-MOBO attainment band and high-margin designs"); ax.set_xlabel("PAE (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=FS["legend"])
    for a in axes:
        a.set_xlim(20, 45.5)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_attainment")


# ---------------------------------------------------------------- selected design across corners
SPEC = {"OP1dB_dBm": (">=", 15.5), "Gain_dB": (">=", 20.0), "minKf": (">=", 5.0), "VDS_peak_M3_V": ("<=", 8.7), "PDC_total_mW": ("<=", 350.0)}


def table_selected_design() -> str:
    from make_tables import corner_designs
    _, per_arm, pooled = corner_designs()
    lhs = load_hist("qn", SEEDS[0]).iloc[:100]
    fb = lhs[feasible_mask(lhs)].sort_values("PAE_percent", ascending=False).iloc[0]
    base_sig = "|".join(str(round(v, 6)) if isinstance(v, float) else str(v) for v in fb[KEY])
    fa_tt = per_arm[(per_arm.arm == MAIN) & (per_arm.corner == "TT/27C") & (per_arm.tt_K >= 15)].sort_values("PAE_percent", ascending=False)
    sel_sig = fa_tt.sig.iloc[0]; sel_seed = int(fa_tt.seed.iloc[0])
    corners = ["TT/27C", "FF/27C", "SS/27C", "FF/-40C", "SS/85C"]
    rows = {}
    for name, sig in (("base", base_sig), ("sel", sel_sig)):
        d = pooled[pooled.sig == sig].set_index("corner")
        assert len(d) == 5, f"{name} not found in corner sweep ({len(d)} rows)"
        rows[name] = d
    metrics = [("PAE_percent", "$PAE$ (\\%)", "--", 1), ("Psat_dBm", "$P_{\\mathrm{sat}}$ (dBm)", "--", 2), ("Gain_dB", "Gain (dB)", "$\\geq$ 20", 1),
               ("OP1dB_dBm", "$OP_{1dB}$ (dBm)", "$\\geq$ 15.5", 2), ("VDS_peak_M3_V", "$V_{\\mathrm{DS,peak}}^{\\mathrm{M3}}$ (V)", "$\\leq$ 8.7", 2),
               ("PDC_total_mW", "$P_{\\mathrm{DC}}$ (mW)", "$\\leq$ 350", 0), ("minKf", "$K$", "$\\geq$ 5", 1)]
    def cell(col, v, dec):
        s = f"{v:.{dec}f}"
        if col in SPEC:
            op, lim = SPEC[col]; bad = (v < lim) if op == ">=" else (v > lim)
            if bad: s = "\\textbf{" + s + "}"
        return s
    lab = lambda c: c.replace("C", "\\,$^\\circ$C").replace("/-", " $-$").replace("/", " ")
    sb = rows["base"].loc["TT/27C"]; ss = rows["sel"].loc["TT/27C"]
    dv = lambda r: (f"$L_x$ library entry {r['Lx_id'].replace('Lx_', '')}, $L_{{\\mathrm{{dc3}}}}$ entry {r['Ldc3_id'].replace('Ldc3_', '')}, "
                    f"$W_{{\\mathrm{{M3}}}}={r['M3_width']:.0f}$\\,$\\mu$m")
    L = [r"\begin{table}[pos=htbp]", r"\centering",
         r"\caption{Selected designs at the nominal and four process--temperature corners with fixed bias. Baseline: the manual baseline design (the best strictly feasible design of the shared seed; " + dv(sb) + r"). FA-MOBO: the highest-$PAE$ FA-MOBO design with nominal $K\geq15$ among the corner-swept designs (seed " + str(sel_seed) + r", augmentation phase; " + dv(ss) + r"). FA-MOBO runs also produced high-margin designs with higher nominal $PAE$ that were not in the corner sweep (Section~\ref{sec:corners}). Bold: specification not met at that corner.}",
         r"\label{tab:selected_design}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{@{}llccccc|ccccc@{}}", r"\toprule",
         r" & & \multicolumn{5}{c|}{Manual baseline} & \multicolumn{5}{c}{FA-MOBO} \\",
         r"Metric & Spec. & " + " & ".join(lab(c) for c in corners) + " & " + " & ".join(lab(c) for c in corners) + r" \\", r"\midrule"]
    for col, name, spec, dec in metrics:
        L.append(f"{name} & {spec} & " + " & ".join(cell(col, rows['base'].loc[c, col], dec) for c in corners) + " & " + " & ".join(cell(col, rows['sel'].loc[c, col], dec) for c in corners) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    info = {"baseline_sig": base_sig, "selected_sig": sel_sig, "selected_seed": sel_seed,
            "baseline_tt": {k: float(sb[k]) for k in ("PAE_percent", "Psat_dBm", "OP1dB_dBm", "minKf")},
            "selected_tt": {k: float(ss[k]) for k in ("PAE_percent", "Psat_dBm", "OP1dB_dBm", "minKf")},
            "selected_ss85_pae": float(rows["sel"].loc["SS/85C", "PAE_percent"]), "baseline_ss85_pae": float(rows["base"].loc["SS/85C", "PAE_percent"]),
            "selected_min_K": float(rows["sel"].minKf.min()), "baseline_min_K": float(rows["base"].minKf.min())}
    import json; (OUT / "selected_design.json").write_text(json.dumps(info, indent=1))
    return "\n".join(L)


# ---------------------------------------------------------------- settings table (values read from code)
def table_settings() -> str:
    """Optimizer settings table of the manuscript.

    In the original workflow this table is read from the optimizer implementation (BoTorch model and acquisition
    defaults, the in-loop classifier and the augmentation settings). That code embeds the foundry PDK inductor
    library, which cannot be distributed, so the generated table is shipped as data/precomputed/tab_settings.tex.
    """
    return (DATA / "precomputed" / "tab_settings.tex").read_text().rstrip("\n")


if __name__ == "__main__":
    fig_attainment()
    (OUT / "tab_selected_design.tex").write_text(table_selected_design() + "\n"); print("wrote tab_selected_design.tex")
    (OUT / "tab_settings.tex").write_text(table_settings() + "\n"); print("wrote tab_settings.tex")
