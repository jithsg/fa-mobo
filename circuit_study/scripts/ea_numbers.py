"""Single-run evolutionary baselines (NSGA-II, GDE3, SMS-EMOA; data/evolutionary/history_*.csv), scored like the 10-seed arms."""
import json
import pandas as pd
from common import BUDGET, DATA, OUT, margin_rich
from problem import feasible_mask
from base import dominated_hypervolume as hv
out = {}
for m in ("NSGA-II", "GDE3", "SMS-EMOA"):
    h = pd.read_csv(DATA / "evolutionary" / f"history_{m}.csv")
    for n in (BUDGET, len(h)):
        d = h.iloc[:n]; fm = feasible_mask(d); mr = margin_rich(d)
        out[f"{m}@{n}"] = {"sims": int(n), "hv": float(hv(d.loc[fm, "PAE_percent"].to_numpy(), d.loc[fm, "Psat_dBm"].to_numpy())),
                           "feasible": int(fm.sum()), "k15": int(mr.sum()), "best_pae": float(d.loc[fm, "PAE_percent"].max()) if fm.any() else None,
                           "unique_feasible": int(d.loc[fm, ["Lx_id", "Ldc3_id", "M3_width", "CB", "Cs", "Cm"]].round(6).drop_duplicates().shape[0])}
(OUT / "ea_numbers.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
