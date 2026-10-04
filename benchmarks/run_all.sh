#!/usr/bin/env bash
# Reproduce every benchmark table of the manuscript from the shipped result records.
# Usage: bash run_all.sh      (set PYTHON=/path/to/python to choose an interpreter)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
mkdir -p results_all && cp -n results/*.json results_all/ 2>/dev/null || true
echo "== aggregating 640 runs"
"$PY" bench_report.py report
echo "== benchmark tables"
"$PY" bench_tables.py
echo "== the circuit on the same axis (needs the circuit data of 04_code_and_data)"
"$PY" make_axis_table.py || echo "   skipped: circuit histories not present"
echo "Tables written to out/tables"
