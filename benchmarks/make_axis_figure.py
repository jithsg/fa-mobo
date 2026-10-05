"""Figure version of the decomposition table: the FA-MOBO - qNEHVI feasible-count difference split into its two terms.

  python make_axis_figure.py          # after bench_report.py report; writes out/fig_axis.png / .pdf

Uses exactly the rows of make_axis_table.py (eight benchmarks + the Class-E PA), ordered by the control BO rate.
Bars: reallocated-evaluation term and subsequent-phase term (positive parts stacked right of zero, negative parts left).
Diamond: measured mean difference. The identity is exact, so the diamond sits at the sum of the two terms.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import make_axis_table as mat  # noqa: E402

C_REALLOC, C_SUBSEQ = "#009E73", "0.25"   # augmentation-phase and BO-phase colours of the circuit figures
NAMES = {"C2DTLZ2": "C2-DTLZ2", "C3DTLZ4": "C3-DTLZ4"}


def main() -> None:
    rows = sorted(mat.benchmark_rows() + [mat.pa_row()], key=lambda r: r["control_bo"])
    for r in rows:
        assert abs(r["direct"] + r["indirect"] - r["measured"]) < 0.01, r["name"]   # exact identity
    import common                            # imported by pa_row(); same house style as the circuit figures
    common.style()
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ys = np.arange(len(rows))[::-1]          # lowest control rate at the top
    h = 0.58
    for y, r in zip(ys, rows):
        pos = neg = 0.0
        for v, c in ((r["direct"], C_REALLOC), (r["indirect"], C_SUBSEQ)):
            left = pos if v >= 0 else neg + v
            ax.barh(y, abs(v), left=left, height=h, color=c, edgecolor="white", linewidth=0.4, zorder=3)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        circ = r["kind"] == "circuit"
        ax.plot(r["measured"], y, marker="D", ms=5.2 if circ else 4.2, mfc="#D55E00" if circ else "white", mec="black", mew=0.8, zorder=5)
        p = "p < 0.001" if r["p"] < 0.0005 else f"p = {r['p']:.3f}"
        ax.text(15.6, y, f"{r['measured']:+.2f}  ({p})", va="center", ha="left", fontsize=7, weight="bold" if circ else "normal",
                color="black" if r["p"] < 0.05 else "0.35")
        if circ:
            ax.axhspan(y - 0.42, y + 0.42, color="#FBE3D6", zorder=0)
    ax.axvline(0, color="0.2", lw=0.7, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{NAMES.get(r['name'], r['name'])}  ({r['control_bo']:.3f})" for r in rows], fontsize=7)
    for t, r in zip(ax.get_yticklabels(), rows):
        if r["kind"] == "circuit":
            t.set_weight("bold")
    ax.set_xlim(-8.6, 15.4); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Feasible designs per run, FA-MOBO minus qNEHVI")
    ax.text(-0.015, len(rows) - 0.4, "problem (qNEHVI BO yield)", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color="0.3")
    ax.text(15.6, len(rows) - 0.4, "measured", ha="left", va="bottom", fontsize=7, color="0.3")
    ax.grid(axis="x", color="0.9", lw=0.5, zorder=0); ax.tick_params(axis="y", length=0)
    handles = [Patch(color=C_REALLOC, label="reallocated evaluations (augmentation batch)"),
               Patch(color=C_SUBSEQ, label="subsequent-phase effect (later acquisition)"),
               Line2D([], [], ls="", marker="D", mfc="white", mec="black", ms=4.2, label="measured difference")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=3, fontsize=7, frameon=False, handlelength=1.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.2, right=0.82, top=0.86, bottom=0.14)
    out = mat.HERE / "out"; out.mkdir(exist_ok=True)
    fig.savefig(out / "fig_axis.png", facecolor="white"); fig.savefig(out / "fig_axis.pdf")
    fig.savefig(out / "fig_axis_preview.png", dpi=110, facecolor="white")
    print("wrote out/fig_axis.png, .pdf, _preview.png")


if __name__ == "__main__":
    main()
