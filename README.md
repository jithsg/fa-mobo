# FA-MOBO: feasibility-augmented, classifier-guided multi-objective Bayesian optimization

Code and data accompanying the article *Feasibility-Augmented Classifier-Guided Multi-Objective
Bayesian Optimization for Sparse-Feasibility Engineering Design: Application to a Class-E CMOS
Power Amplifier*, by Jithish Jayarajan, Yuqing Liu, Bharatha Kumar Thangarasu, Nagarajan
Mahalingam, Xuehua Li, Yujie Li and Kiat Seng Yeo, under review at *Applied Soft Computing*.

FA-MOBO targets constrained
multi-objective design problems in which only a small fraction of candidate designs is feasible.
It simulates a small feasibility-augmentation batch ranked by bootstrap random-forest ensembles,
then guides constrained qLogNEHVI with an in-loop feasibility classifier.

The repository has two independent parts.

## `benchmarks/` — fully reproducible, no proprietary dependency

A study of FA-MOBO and three ablations (Aug-BO, Clf-BO, plain qNEHVI) on eight constrained
multi-objective benchmark problems — MW2, MW3, MW7, MW11, C2-DTLZ2, C3-DTLZ4, TNK and OSY —
with 20 paired seeds per arm and problem (640 runs). Everything needed to regenerate every
benchmark table of the article is here:

- `bench_problems.py` — the eight problems in pure numpy, checked against `pymoo` to the last
  bit by `validate_against_pymoo.py`
- `bench_arms.py`, `bench_run.py` — the four arms and the runner
- `results/` — all 640 run records (per-evaluation objectives, constraints, phase, feasibility)
- `bench_report.py`, `bench_tables.py`, `make_axis_table.py` — aggregation and tables
- `make_axis_figure.py` — the main-text decomposition figure (its values are Supplementary Table S8)
- `pre_registration.json`, `pre_registration_outcome.json` — a prediction recorded and
  hash-pinned before two of the problems were run, and its scored outcome
- `bench_targets.json` — the fixed hypervolume targets

      cd benchmarks && pip install -r requirements.txt && bash run_all.sh

`make_axis_table.py` and `make_axis_figure.py` place the circuit study on the same axis as the benchmarks and therefore
read `circuit_study/`; run the circuit part first, or it is skipped with a message.

## `circuit_study/` — the Class-E power-amplifier study

Run histories and analysis scripts for the circuit study: ten paired seeds of five Bayesian
arms, four evolutionary baselines, the 100-design Latin-hypercube seed, the corner sweep, and
every script that produces the article's circuit figures and tables, including the
random-forest classifier analysis (`rf_classifier_analysis.py`: model selection, probability
calibration, adaptive refitting, permutation importance and ensemble disagreement).

      cd circuit_study && pip install -r requirements.txt && bash run_all.sh

Outputs are written to `circuit_study/outputs/`. The full augmentation sensitivity sweep is
skipped by default (`make_sensitivity.py --table-only`); run `python make_sensitivity.py` for
the complete sweep, about 90 minutes.

**What is deliberately not here, and why.** The circuit is a 2.4 GHz Class-E PA in a
commercial 40 nm CMOS process. The foundry process design kit — the spiral-inductor library
(inductance, Q and self-resonance of each entry), the device models, and the netlists that
reference them — is covered by a non-disclosure agreement and cannot be distributed. Three
consequences follow:

- The two inductor design variables appear in every data file as opaque library indices
  (`Lx_10`, `Ldc3_1`, ...), never as electrical values. The encoding and snap-to-library code that
  maps them is omitted; `circuit_study/scripts/problem.py` is an excerpt that keeps the public
  bounds, objectives, constraints and the feasibility test.
- The harmonic-balance evaluator cannot be run from this repository, so the circuit study is
  reproducible at the level of its analysis (every figure and table from the shipped histories)
  but not at the level of new simulations. The benchmark study is the independently
  reproducible part of the work and exists for exactly that reason.
- The in-loop classifier was run on a 10-dimensional encoding in which each inductor contributes
  three component properties (value, quality factor, self-resonance). That encoding needs the
  PDK library, so a fresh run here falls back to the library-index encoding and reports those
  rows only. The article reports both; the component-property numbers it quotes are in
  `circuit_study/data/precomputed/rf_classifier_numbers.json`, so they can be checked without
  the PDK. To rebuild the tables and figures from those stored numbers instead of recomputing:

      mkdir -p outputs/classifier_validation
      cp data/precomputed/rf_classifier_numbers.json outputs/classifier_validation/numbers.json
      cd scripts && python rf_classifier_analysis.py --text-only     # tables and section text
      cd scripts && python rf_classifier_analysis.py --figures-only  # figures

## Relation to the article's tables and figures

The scripts regenerate the body of every table; the captions in the published article were
edited for length and readability after generation, and some tables were re-laid out (for
example a metric column folded into `\multirow`), so a regenerated `.tex` will not be
byte-identical to the typeset one. The numbers are.

A few artefacts are not produced by this code and are not included: the circuit schematic, the
prior-work comparison table, and the summary of average benchmark ranks, which is a transcription
of the per-problem ranks in `tab_bench_ranks_*`.

Paired-test p-values below 0.001 are rendered here as `p<0.001` (the smallest is about
1.2e-4); the article's tables print them as `0.001`.

File names differ slightly between the scripts and the article: `tab_rf_comparison`,
`tab_rf_calibration` and `fig_rf_feature_importance` appear there as `tab_s_rf_models`,
`tab_s_rf_calibration` and `fig_s_rf_feature_importance`.

## Environment

Tested with Python 3.10. Pinned versions are in each part's `requirements.txt`; the benchmark
arms need BoTorch 0.12 and PyTorch, the circuit analysis only numpy, pandas, scipy,
matplotlib and scikit-learn.

## Citing this work

Please cite the article. Until it appears, cite this repository (https://github.com/jithsg/fa-mobo);
`CITATION.cff` carries the metadata and GitHub renders a ready-made citation from it. An archived
release with a DOI will be deposited on Zenodo when the article is accepted, and the DOI added here.

## License

Code is released under the MIT License (see `LICENSE`). The data files are released for
research use with the article; please cite it if you use them.
