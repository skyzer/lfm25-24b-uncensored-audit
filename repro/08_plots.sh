#!/usr/bin/env bash
# All charts in ../plots and ../thread/img. Reads results/ (or ../expected_outputs with RES=../expected_outputs).
set -euo pipefail; . venv/bin/activate
export RES=${RES:-results}
python scripts/proof_png.py
python scripts/plots.py
