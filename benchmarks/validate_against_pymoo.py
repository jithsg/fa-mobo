"""Check every problem in bench_problems against its pymoo 0.6.2 definition (run where pymoo is installed)."""
import numpy as np
from pymoo.problems import get_problem

from bench_problems import PROBLEMS

rng = np.random.default_rng(0)
ok = True
for name, p in PROBLEMS.items():
    ref = get_problem(p.pymoo_name, n_var=p.n_var) if p.pymoo_name.startswith("mw") else get_problem(p.pymoo_name)
    X = p.xl + rng.random((500, p.n_var)) * (p.xu - p.xl)
    F, G = p.evaluate(X)
    out = ref.evaluate(X, return_values_of=["F", "G"])
    Fr, Gr = out[0], out[1]
    dF = float(np.nanmax(np.abs(F - Fr))); dG = float(np.nanmax(np.abs(G - Gr)))
    feas = float(np.mean((G <= 0).all(axis=1)))
    same_shape = F.shape == Fr.shape and G.shape == Gr.shape
    good = same_shape and dF < 1e-9 and dG < 1e-9
    ok &= good
    print(f"{name:5s} n_var={p.n_var:2d} n_constr={p.n_constr} | max|dF|={dF:.2e} max|dG|={dG:.2e} | "
          f"random-sample feasibility {feas:5.1%} | {'MATCH' if good else 'MISMATCH'}")
print("all problems match pymoo" if ok else "MISMATCHES FOUND")
