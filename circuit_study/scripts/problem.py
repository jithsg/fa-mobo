"""Constrained Class-E PA problem definition used by the analysis scripts.

Excerpt of the optimization code: design-variable bounds, objectives, constraints and the strict feasibility
test. The PDK inductor library (inductance, Q and SRF of each spiral) and the encoding/snapping code that uses
it are omitted because the foundry non-disclosure agreement does not allow their distribution.
"""

from __future__ import annotations

import pandas as pd

# --- Continuous design variables and bounds (Table 3 / notebook) -------------
CONTINUOUS_BOUNDS: dict[str, tuple[float, float]] = {
    "M3_width": (640.0, 1280.0),
    "CB": (2.0, 34.0),
    "Cs": (4.0, 33.0),
    "Cm": (4.0, 38.0),
}
CONTINUOUS_VARS: tuple[str, ...] = tuple(CONTINUOUS_BOUNDS)

# --- Objectives --------------------------------------------------------------
OBJECTIVES: tuple[str, str] = ("PAE_percent", "Psat_dBm")
HV_REF_POINT: tuple[float, float] = (0.0, 0.0)  # (PAE%, Psat dBm), maximization

# --- Constraints (Table 6) ---------------------------------------------------
CONSTRAINTS: dict[str, float] = {
    "OP1dB_dBm_min": 15.5,
    "Gain_dB_min": 20.0,
    "minKf_min": 5.0,
    "VDS_peak_M3_V_max": 8.7,
    "PDC_total_mW_max": 350.0,
}

# --- I/O contract: columns the remote-workstation Spectre flow consumes ------
DESIGN_COLS: tuple[str, ...] = ("Lx_id", "Ldc3_id", "M3_width", "CB", "Cs", "Cm")
METRIC_COLS: tuple[str, ...] = (
    "PAE_percent", "Psat_dBm", "OP1dB_dBm", "Gain_dB",
    "PDC_total_mW", "VDS_peak_M3_V", "minKf", "HB_converged_flag",
)


def feasible_mask(df: pd.DataFrame) -> pd.Series:
    """Strict feasibility per Table 6 (boolean Series aligned to ``df``).

    A design is feasible iff it satisfies all six constraints. Missing metric
    values (e.g. unsimulated rows) are treated as infeasible.
    """
    cols = ["OP1dB_dBm", "Gain_dB", "minKf", "VDS_peak_M3_V", "PDC_total_mW", "HB_converged_flag"]
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            raise KeyError(f"feasible_mask requires column '{c}'")
    ok = (
        (d["OP1dB_dBm"] >= CONSTRAINTS["OP1dB_dBm_min"])
        & (d["Gain_dB"] >= CONSTRAINTS["Gain_dB_min"])
        & (d["minKf"] >= CONSTRAINTS["minKf_min"])
        & (d["VDS_peak_M3_V"] <= CONSTRAINTS["VDS_peak_M3_V_max"])
        & (d["PDC_total_mW"] <= CONSTRAINTS["PDC_total_mW_max"])
        & (d["HB_converged_flag"] >= 0.5)
    )
    return ok.fillna(False)
