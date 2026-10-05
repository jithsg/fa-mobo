"""Forest plot of the paired comparison of FA-MOBO against each Bayesian arm (figure version of tab_ms_tests).

Output: outputs/fig_tests.png / .pdf / _preview.png. Run from the scripts folder: python make_fig_tests.py
Numbers come from make_tables.t3_numbers (same bootstrap seed and tests as the table); the script stops if the
table rows it regenerates differ from the manuscript's tab_ms_tests.tex (when that file is present).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import make_tables as mt  # noqa: E402
from common import ARMS, ROOT, save, style  # noqa: E402

style()
OTHERS = mt.OTHERS
SIG = 0.05
# (key, panel title, decimals); joint-target panels use counts (positive = favours FA-MOBO)
CONT = [("feasible", "Strictly feasible designs", 1), ("high_margin", "High-margin designs", 1),
        ("hv3", "Margin-aware hypervolume", 0), ("hv2", "2-D hypervolume", 0),
        ("best_pae", "Best PAE (%)", 1), ("best_pae_hm", "Best PAE, high-margin (%)$^\\dagger$", 1)]
MS_TESTS_TEX = ROOT.parent.parent / "Manuscript" / "LaTeX_Source" / "tables" / "tab_ms_tests.tex"


def p_text(p: float) -> str:
    return "p < 0.001" if p < 0.0005 else f"p = {p:.3f}"


def check_against_table() -> None:
    """Regenerate the table rows and compare them with the manuscript copy (rows only; captions are edited by hand)."""
    rows_new = [ln for ln in mt.table_tests().splitlines() if "&" in ln and not ln.startswith("Outcome")]
    src = next((p for p in (MS_TESTS_TEX, ROOT.parent.parent / "Supplementary" / "LaTeX_Source" / "tables" / "tab_s_ms_tests.tex") if p.exists()), None)
    if src is None:
        print("table check skipped: no tab_ms_tests copy found"); return
    rows_old = [ln for ln in src.read_text(encoding="utf-8").splitlines() if "&" in ln and not ln.startswith("Outcome")]
    norm = lambda s: s.split("&", 1)[1].replace(" ", "")  # noqa: E731  (the outcome label may be reworded)
    if [norm(r) for r in rows_new] != [norm(r) for r in rows_old]:
        raise SystemExit(f"table check FAILED against {src}")
    print(f"table check: {len(rows_new)} rows identical to {src.name}")


def forest(ax, vals, title: str, d: int, first: bool, xlabel: str) -> None:
    ys = np.arange(len(OTHERS))[::-1]
    for y, a, v in zip(ys, OTHERS, vals):
        c = ARMS[a][2]; sig = v["p"] < SIG
        ax.plot(v["ci95"], [y, y], color=c, lw=1.4, solid_capstyle="butt", zorder=3)
        ax.plot(v["diff"], y, "o", ms=4.6, mfc=c if sig else "white", mec=c, mew=1.0, zorder=4)
        ax.text(1.02, y, p_text(v["p"]), transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.5,
                color="black" if sig else "0.45", clip_on=False)
    _frame(ax, ys, title, first, xlabel)


def counts(ax, pos, neg, ps, title: str, first: bool, xlabel: str) -> None:
    ys = np.arange(len(OTHERS))[::-1]
    for y, a, n_pos, n_neg, p in zip(ys, OTHERS, pos, neg, ps):
        c = ARMS[a][2]; sig = p < SIG
        ax.barh(y, n_pos, height=0.5, color=c if sig else "white", edgecolor=c, lw=1.0, zorder=3)
        if n_neg:
            ax.barh(y, -n_neg, height=0.5, color="white", edgecolor=c, lw=1.0, hatch="////", zorder=3)
        ax.text(n_pos + 0.3, y, f"{n_pos}/{n_neg}", va="center", ha="left", fontsize=6.5, color="0.25")
        ax.text(1.02, y, p_text(p), transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.5,
                color="black" if sig else "0.45", clip_on=False)
    ax.set_xlim(-2, 16); ax.set_xticks([0, 5, 10])
    _frame(ax, ys, title, first, xlabel)


def _frame(ax, ys, title, first, xlabel) -> None:
    ax.axvline(0, color="0.3", lw=0.6, zorder=1)
    ax.set_yticks(ys); ax.set_yticklabels([f"vs {ARMS[a][0]}" for a in OTHERS] if first else [])
    ax.tick_params(axis="y", length=0); ax.set_ylim(-0.6, len(OTHERS) - 0.4)
    ax.set_title(title, fontsize=7.5); ax.set_xlabel(xlabel, fontsize=7)
    ax.grid(axis="x", color="0.92", lw=0.5, zorder=0)


def main() -> None:
    t3 = mt.t3_per_run()
    num = mt.t3_numbers(t3); mt.NUM["t3"] = num
    check_against_table()
    vs = num["vs"]
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.7))
    letters = iter("abcdefgh")
    layout = [("cont", CONT[0]), ("cont", CONT[1]), ("reached", None), ("sooner", None),
              ("cont", CONT[2]), ("cont", CONT[3]), ("cont", CONT[4]), ("cont", CONT[5])]
    for i, (ax, (kind, spec)) in enumerate(zip(axs.flat, layout)):
        first = i % 4 == 0; L = next(letters)
        if kind == "cont":
            key, title, d = spec
            forest(ax, [vs[a][key] for a in OTHERS], f"({L}) {title}", d, first, "FA-MOBO minus arm")
        elif kind == "reached":
            m = [vs[a]["joint_target_mcnemar"] for a in OTHERS]
            counts(ax, [x["fa_only"] for x in m], [x["arm_only"] for x in m], [x["p"] for x in m],
                   f"({L}) Joint target reached", first, "seeds: FA-MOBO only / arm only")
        else:
            s = [vs[a]["joint_target_sooner"] for a in OTHERS]
            counts(ax, [x["sooner"] for x in s], [x["later"] for x in s], [x["p"] for x in s],
                   f"({L}) Joint target sooner", first, "seeds: sooner / later")
    fig.subplots_adjust(left=0.1, right=0.955, top=0.93, bottom=0.12, wspace=0.62, hspace=0.62)
    save(fig, "fig_tests")


if __name__ == "__main__":
    main()
