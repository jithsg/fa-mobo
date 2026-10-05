"""Fig. 1: FA-MOBO pipeline diagram, drawn at print width 7.0 in.

Every number in the white strips is FA-MOBO's mean per run over the ten seeds, computed from the run histories
(common.per_seed_table). Replaces fig_architecture() in make_figures.py, whose feedback loop was drawn as three
separate arrow patches with shrunken ends and so rendered with gaps. Run from the scripts folder:
    python make_fig_architecture.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

from common import (BOX_COLORS, BUDGET, DESIGN_VARS, FS, N_CONSTRAINTS, N_SEED, OUT, Q_BATCH,  # noqa: E402
                    per_seed_table, save, style)

MAIN = "fa15qc"
W, H = 7.0, 2.55                 # figure size (in)
XMAX, YMAX = 100.0, 30.0         # data units of the drawing canvas
PAD = 0.4                        # FancyBboxPatch pad: the drawn edge lies PAD outside the nominal rectangle
GAP = 5.2                        # horizontal gap between nominal rectangles
Y0, BOX_H = 7.4, 22.0            # nominal bottom and height of the four phase boxes
STRIP_H = 5.4                    # height of the white outcome strip
INK = "0.15"
LW = 0.9
NOTE = FS["note"]


def outcome_numbers() -> dict:
    """FA-MOBO means per run: feasible and high-margin designs by phase and in total."""
    ps = per_seed_table()
    fa = ps[ps["arm"] == MAIN]
    return {"n_aug": int(fa["n_aug"].iloc[0]), "n_opt": int(fa["n_opt"].iloc[0]),
            "seed_feasible": float(fa["feasible_lhs"].mean()), "seed_high_margin": float(fa["k15_lhs"].mean()),
            "aug_feasible": float(fa["feasible_aug"].mean()), "aug_high_margin": float(fa["k15_aug"].mean()),
            "bo_feasible": float(fa["feasible_opt"].mean()), "bo_high_margin": float(fa["k15_opt"].mean()),
            "feasible": float(fa["feasible"].mean()), "high_margin": float(fa["k15"].mean())}


def box(ax, x: float, w: float, phase: str, title: str, body: str, strip: str, fc: str) -> None:
    top = Y0 + BOX_H
    ax.add_patch(FancyBboxPatch((x, Y0), w, BOX_H, boxstyle=f"round,pad={PAD},rounding_size=1.2", fc=fc, ec="0.3", lw=0.7))
    ax.text(x + w / 2, top - 1.0, phase, ha="center", va="top", fontsize=FS["title"], weight="bold")
    ax.text(x + w / 2, top - 4.3, title, ha="center", va="top", fontsize=NOTE, style="italic")
    ax.text(x + w / 2, top - 7.9, body, ha="center", va="top", fontsize=NOTE, linespacing=1.28)
    ax.add_patch(FancyBboxPatch((x + 1.0, Y0 + 0.8), w - 2.0, STRIP_H, boxstyle="round,pad=0.2,rounding_size=0.6", fc="white", ec="0.6", lw=0.5))
    ax.text(x + w / 2, Y0 + 0.8 + STRIP_H / 2, strip, ha="center", va="center", fontsize=NOTE, color=INK, linespacing=1.25)


def arrow(ax, x_edge0: float, x_edge1: float, y: float, label: str) -> None:
    """Straight arrow from one drawn box edge to the next (no end shrinking, so it touches both edges)."""
    ax.add_patch(FancyArrowPatch((x_edge0, y), (x_edge1, y), arrowstyle="-|>", mutation_scale=9, lw=LW, color=INK, shrinkA=0, shrinkB=0))
    ax.text((x_edge0 + x_edge1) / 2, y + 1.0, label, ha="center", va="bottom", fontsize=NOTE, weight="bold")


def feedback_loop(ax, x_out: float, x_in: float, y_edge: float, y_low: float) -> None:
    """One continuous polyline leaving the box bottom, running below it and re-entering it, with a single arrowhead."""
    path = Path([(x_out, y_edge), (x_out, y_low), (x_in, y_low), (x_in, y_edge)], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=9, lw=LW, color=INK, shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt"))


def fig_architecture() -> dict:
    style()
    n = outcome_numbers()
    widths = [19.8, 22.6, 25.0, 15.6]
    xs = [0.6]
    for w in widths[:-1]:
        xs.append(xs[-1] + w + GAP)
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, XMAX); ax.set_ylim(0, YMAX); ax.axis("off")
    fig.subplots_adjust(left=0, right=0.975, bottom=0, top=1)   # saved width ~6.86 in as for the other figures: 7 pt prints at ~6.6 pt

    box(ax, xs[0], widths[0], "Phase I", "Space-filling seed",
        f"{N_SEED}-design Latin hypercube\nover {len(DESIGN_VARS)} design variables;\ninductors chosen from a\ncharacterized component library",
        f"{n['seed_feasible']:.0f} of {N_SEED} feasible\n{n['seed_high_margin']:.0f} high-margin", BOX_COLORS["seed"])
    box(ax, xs[1], widths[1], "Phase II", "Feasibility-augmented sampling",
        f"Random-forest feasibility\nensembles rank candidates by\nP(feasible) × efficiency weight;\nsimulate the top {n['n_aug']} (Alg. 1)",
        f"{n['aug_feasible']:.1f} of {n['n_aug']} feasible\n{n['aug_high_margin']:.1f} high-margin", BOX_COLORS["aug"])
    box(ax, xs[2], widths[2], "Phase III", "Classifier-guided constrained MOBO",
        f"qLogNEHVI on GPs of PAE, Psat\nand {N_CONSTRAINTS} constraints; log-acquisition\nplus log P(feasible) of an in-loop\nrandom-forest classifier",
        f"{n['bo_feasible']:.1f} of {n['n_opt']} feasible\n{n['bo_high_margin']:.1f} high-margin", BOX_COLORS["bo"])
    box(ax, xs[3], widths[3], "Output", "convergence-informed\nevaluation budget",
        f"Pareto set in\n(PAE, Psat) and\nhigh-margin\ndesigns",
        f"{n['feasible']:.1f} feasible\n{n['high_margin']:.1f} high-margin", BOX_COLORS["out"])

    y_mid = Y0 + BOX_H * 0.55
    counts = [N_SEED, N_SEED + n["n_aug"], BUDGET]
    for i, c in enumerate(counts):
        arrow(ax, xs[i] + widths[i] + PAD, xs[i + 1] - PAD, y_mid, f"{c}")

    # Phase III feedback loop: out of the box bottom, below it, and back in
    x3, w3 = xs[2], widths[2]
    y_edge, y_low = Y0 - PAD, 3.1
    x_out, x_in = x3 + w3 * 0.78, x3 + w3 * 0.22
    feedback_loop(ax, x_out, x_in, y_edge, y_low)
    ax.text(x_out + 1.4, y_low + 0.1, f"after every batch of {Q_BATCH}: refit the\nGPs and the classifier on all designs",
            ha="left", va="center", fontsize=NOTE, color="0.25", linespacing=1.2)
    ax.text(xs[0] - PAD, y_low + 0.1, "Numbers on arrows: cumulative simulations\nWhite strips: FA-MOBO mean per run, ten seeds",
            ha="left", va="center", fontsize=NOTE, color="0.25", linespacing=1.2)

    save(fig, "fig_architecture")
    plt.close(fig)
    return n


if __name__ == "__main__":
    nums = fig_architecture()
    (OUT / "fig_architecture_numbers.json").write_text(json.dumps(nums, indent=1))
    print(json.dumps(nums))
