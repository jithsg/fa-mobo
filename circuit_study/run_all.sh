#!/usr/bin/env bash
# Regenerate every figure, table and quoted number of the manuscript and supplement from data/.
# Usage: bash run_all.sh            (uses python3; set PYTHON=/path/to/python to choose another interpreter)
set -euo pipefail
cd "$(dirname "$0")/scripts"
PY=${PYTHON:-python3}
export MPLBACKEND=Agg
for s in make_tables.py make_fig_tests.py make_figures.py make_fig_architecture.py make_extra.py make_gabstract.py make_novel.py make_novel_tables.py ea_numbers.py rf_classifier_analysis.py; do
  echo "== $s"
  "$PY" "$s"
done
echo "== make_sensitivity.py --table-only   (full sweep: python make_sensitivity.py, about 90 min)"
"$PY" make_sensitivity.py --table-only
echo "All outputs written to $(cd .. && pwd)/outputs"
