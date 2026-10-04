"""Write the pre-registration for the C-DTLZ extension, before any design seed is run.

  python make_preregistration.py            # writes pre_registration.json

The prediction rule was derived post hoc from the first six problems (see bench_report.py).
This file records it as a forward prediction for two problems it has never seen, estimated
only from pilot runs at seeds outside the 101-120 design set, so the design runs are a
genuine out-of-sample test rather than a restatement of the fit.
"""
from __future__ import annotations

import collections
import glob
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW_PROBLEMS = ("C2DTLZ2", "C3DTLZ4")
DESIGN_SEEDS = list(range(101, 121))
CONTROL = "qNEHVI"


def phase_counts(recs: list[dict], phase: str) -> tuple[int, int]:
    """Feasible designs found, and evaluations spent, in one phase across the given runs."""
    found = spent = 0
    for r in recs:
        for ph, fe in zip(r.get("phase", []), r["feasible"]):
            if ph == phase:
                spent += 1
                found += bool(fe)
    return found, spent


def main() -> None:
    by: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    pilot_seeds: set[int] = set()
    for f in sorted(glob.glob(str(HERE / "pilot" / "*.json"))):
        r = json.loads(Path(f).read_text())
        by[(r["problem"], r["arm"])].append(r)
        pilot_seeds.add(r["seed"])
    if pilot_seeds & set(DESIGN_SEEDS):
        raise SystemExit(f"pilot used design seeds {sorted(pilot_seeds & set(DESIGN_SEEDS))}; that would not be out of sample")

    predictions = {}
    for prob in NEW_PROBLEMS:
        fa, ctl = by[(prob, "FA-MOBO")], by[(prob, CONTROL)]
        if not fa or not ctl:
            raise SystemExit(f"{prob}: pilot incomplete (FA-MOBO {len(fa)}, {CONTROL} {len(ctl)})")
        aug_found, aug_spent = phase_counts(fa, "aug")
        bo_found, bo_spent = phase_counts(ctl, "bo")
        aug_rate = aug_found / aug_spent
        bo_rate = bo_found / bo_spent
        n_aug = collections.Counter(fa[0]["phase"])["aug"]
        predicted = n_aug * (aug_rate - bo_rate)
        predictions[prob] = {
            "augmentation_rate": aug_rate, "augmentation_trials": aug_spent,
            "bo_rate_of_control": bo_rate, "bo_trials": bo_spent,
            "augmentation_evaluations": n_aug,
            "predicted_mean_difference_vs_control": predicted,
            "predicted_sign": "FA-MOBO ahead" if predicted > 0 else "FA-MOBO behind",
            "pilot_seeds_used": sorted({r["seed"] for r in fa + ctl}),
        }

    doc = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "written BEFORE any run at seeds 101-120 on these problems",
        "rule": "predicted difference in feasible designs = augmentation_evaluations * (augmentation conversion rate - control BO conversion rate); "
                "a conversion rate is feasible designs found per evaluation spent in that phase",
        "rule_origin": "derived post hoc from the first six problems, where it tracked the measured paired difference with r = 0.971",
        "problems": list(NEW_PROBLEMS),
        "arms": ["FA-MOBO", "Aug-BO", "Clf-BO", CONTROL],
        "design_seeds": DESIGN_SEEDS,
        "pilot_seeds": sorted(pilot_seeds),
        "predictions": predictions,
        "analysis_plan": {
            "primary_metric": "feasible designs per run",
            "comparison": f"FA-MOBO minus {CONTROL}, paired by seed over the 20 design seeds",
            "test": "two-sided Wilcoxon signed-rank on the non-zero paired differences",
            "confirmation": "the sign of the measured mean paired difference matches predicted_sign for each problem",
            "reporting": "both problems are reported whatever the outcome; a failed prediction is reported as a failed prediction",
        },
        "code_state": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                     cwd=str(HERE)).stdout.strip() or "not a git repository",
    }
    out = HERE / "pre_registration.json"
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out}")
    for prob, p in predictions.items():
        print(f"  {prob}: aug {p['augmentation_rate']:.3f} ({p['augmentation_trials']} trials) vs BO {p['bo_rate_of_control']:.3f} "
              f"({p['bo_trials']} trials) -> predicted {p['predicted_mean_difference_vs_control']:+.2f}, {p['predicted_sign']}")


if __name__ == "__main__":
    main()
