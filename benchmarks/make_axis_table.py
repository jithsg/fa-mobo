"""Place the PA circuit on the same conversion-rate axis as the benchmark problems.

  python make_axis_table.py           # prints the table and writes out/tables/tab_axis.tex

Both the circuit and the benchmarks split an equal post-seed budget between an augmentation
phase and a BO phase, on a seed shared by every arm, so the difference in feasible designs
against the plain control decomposes exactly into
    direct   = n_aug * (augmentation rate - control BO rate)
    indirect = n_bo  * (FA-MOBO BO rate   - control BO rate)
This is an accounting identity, computed after the fact; it explains rather than predicts.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
_CANDIDATES = (HERE.parent / "circuit_study" / "scripts",            # released layout: circuit study beside benchmarks/
               HERE.parent / "robustness" / "paper_assets")          # original working tree
PA_ASSETS = next((c for c in _CANDIDATES if (c / "common.py").exists()), _CANDIDATES[0])


def _pcell(p: float) -> str:
    """p-value for a table cell; see _ptex in bench_tables.py."""
    return "$<0.001$" if p < 0.0005 else f"{p:.3f}"


def rate(recs: list[dict], phase: str) -> tuple[float, int]:
    found = spent = 0
    for r in recs:
        for ph, fe in zip(r.get("phase", []), r["feasible"]):
            if ph == phase:
                spent += 1
                found += bool(fe)
    return (found / spent if spent else float("nan")), spent // max(len(recs), 1)


def benchmark_rows() -> list[dict]:
    recs: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for f in glob.glob(str(HERE / "results_all" / "*.json")):
        r = json.loads(Path(f).read_text())
        if r["seed"] in range(101, 121):
            recs[(r["problem"], r["arm"])].append(r)
    tests = json.loads((HERE / "out" / "bench_numbers.json").read_text())["tests"]
    rows = []
    for prob in sorted({p for p, _ in recs}):
        fa, ctl = recs[(prob, "FA-MOBO")], recs[(prob, "qNEHVI")]
        ar, n_aug = rate(fa, "aug")
        fr, n_bo = rate(fa, "bo")
        cr, _ = rate(ctl, "bo")
        t = tests[prob]["feasible"]["qNEHVI"]
        rows.append({"name": prob, "kind": "benchmark", "control_bo": cr, "fa_bo": fr, "aug": ar,
                     "direct": n_aug * (ar - cr), "indirect": n_bo * (fr - cr),
                     "measured": t["diff_mean"], "p": t["p"], "n_aug": n_aug, "n_bo": n_bo})
    return rows


def pa_row() -> dict:
    cwd = os.getcwd()
    os.chdir(PA_ASSETS)
    sys.path.insert(0, str(PA_ASSETS))
    try:
        import common
        tot = {a: {"aug": [0, 0], "opt": [0, 0]} for a in ("fa15qc", "qn")}
        per_seed = {"fa15qc": [], "qn": []}
        for arm in ("fa15qc", "qn"):
            for s in common.SEEDS:
                d = common.load_hist(arm, s).iloc[:common.BUDGET]
                f = np.asarray(common.feasible_mask(d)).astype(bool)
                ph = d["phase"].to_numpy()
                for key in ("aug", "opt"):
                    m = ph == key
                    tot[arm][key][0] += int(f[m].sum())
                    tot[arm][key][1] += int(m.sum())
                per_seed[arm].append(int(f.sum()))
        n = len(common.SEEDS)
        ar = tot["fa15qc"]["aug"][0] / tot["fa15qc"]["aug"][1]
        fr = tot["fa15qc"]["opt"][0] / tot["fa15qc"]["opt"][1]
        cr = tot["qn"]["opt"][0] / tot["qn"]["opt"][1]
        n_aug, n_bo = tot["fa15qc"]["aug"][1] // n, tot["fa15qc"]["opt"][1] // n
        diff = np.array(per_seed["fa15qc"]) - np.array(per_seed["qn"])
        nz = diff[diff != 0]
        return {"name": "Class-E PA", "kind": "circuit", "control_bo": cr, "fa_bo": fr, "aug": ar,
                "direct": n_aug * (ar - cr), "indirect": n_bo * (fr - cr),
                "measured": float(diff.mean()), "p": float(wilcoxon(nz).pvalue) if len(nz) else 1.0,
                "n_aug": n_aug, "n_bo": n_bo}
    finally:
        os.chdir(cwd)


def main() -> None:
    rows = sorted(benchmark_rows() + [pa_row()], key=lambda r: r["control_bo"])
    print(f"{'problem':12s} {'ctl BO':>7s} {'FA BO':>7s} {'gap':>7s} {'direct':>8s} {'indirect':>9s} {'sum':>7s} {'measured':>9s} {'p':>7s}")
    for r in rows:
        star = " *" if r["p"] < 0.05 else ""
        tag = "  <= circuit" if r["kind"] == "circuit" else ""
        print(f"{r['name']:12s} {r['control_bo']:7.3f} {r['fa_bo']:7.3f} {r['fa_bo']-r['control_bo']:+7.3f} "
              f"{r['direct']:+8.2f} {r['indirect']:+9.2f} {r['direct']+r['indirect']:+7.2f} {r['measured']:+9.2f} {r['p']:7.3f}{star}{tag}")

    gaps = np.array([r["fa_bo"] - r["control_bo"] for r in rows])
    meas = np.array([r["measured"] for r in rows])
    print(f"\n  correlation of measured advantage with the BO rate GAP:      r = {np.corrcoef(gaps, meas)[0,1]:.3f}")
    ctl = np.array([r["control_bo"] for r in rows])
    print(f"  correlation of measured advantage with the CONTROL rate:     r = {np.corrcoef(ctl, meas)[0,1]:.3f}")
    ind = np.array([r["indirect"] for r in rows])
    print(f"  indirect share on the circuit: {rows[[r['kind'] for r in rows].index('circuit')]['indirect'] / rows[[r['kind'] for r in rows].index('circuit')]['measured']:.0%}")

    L = [r"\begin{table*}[pos=htbp]", r"\centering",
         r"\caption{The Class-E PA circuit placed on the same axis as the benchmark problems. Every arm shares the seed phase and spends an "
         r"equal post-seed budget, so the mean difference in feasible designs against qNEHVI decomposes exactly into a direct term, "
         r"$n_{\mathrm{aug}}(r_{\mathrm{aug}}-r_{\mathrm{BO}}^{\mathrm{ctl}})$, and an indirect term, "
         r"$n_{\mathrm{BO}}(r_{\mathrm{BO}}^{\mathrm{FA}}-r_{\mathrm{BO}}^{\mathrm{ctl}})$, where a rate is feasible designs found per evaluation "
         r"spent in that phase. The indirect term measures the augmented data improving the feasibility surrogate used by the BO phase. "
         r"Rows are ordered by the control BO rate. The decomposition is exact by construction and was computed after the results were known; "
         r"it explains the outcome rather than predicting it.}",
         r"\label{tab:axis}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Problem & $r_{\mathrm{BO}}^{\mathrm{ctl}}$ & $r_{\mathrm{BO}}^{\mathrm{FA}}$ & Direct & Indirect & Measured & $p$ \\", r"\midrule"]
    for r in rows:
        nm = r"\textbf{" + r["name"] + "}" if r["kind"] == "circuit" else r["name"]
        L.append(f"{nm} & {r['control_bo']:.3f} & {r['fa_bo']:.3f} & {r['direct']:+.2f} & {r['indirect']:+.2f} & "
                 f"{r['measured']:+.2f} & {_pcell(r['p'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out = HERE / "out" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_axis.tex").write_text("\n".join(L) + "\n")
    print(f"\nwrote {out / 'tab_axis.tex'}")


if __name__ == "__main__":
    main()
