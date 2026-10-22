#!/usr/bin/env bash
# Reproduce every result in this repository from scratch.
# Runtime: about 4 minutes on a laptop. No network access required.
set -euo pipefail
cd "$(dirname "$0")/src"
echo "[1/5] synthetic EHR cohort";        python3 generate_ehr.py
echo "[2/5] wearable time-series";        python3 generate_wearables.py
echo "[3/5] two-stream feature fusion";   python3 features.py
echo "[4/5] train + evaluate + ablate";   python3 train_eval.py
echo "[5/5] twin counterfactual validation"; python3 twin.py
echo "[+]   clinician dashboard";         python3 build_dashboard.py
echo "[+]   deployment economics";        python3 deployment.py
echo "[+]   refresh README + PDFs";       python3 report.py
echo
echo "Optional, slower:"
echo "  python3 src/sensitivity.py       coupling sweep, ~3 min"
echo "  python3 src/validate_sharee.py   real-data validation, needs docs/SHAREE_DOWNLOAD.md"
echo "  python3 -m pytest tests/ -q      15 tests"
echo
echo "Done. Open dashboard/index.html and see results/ for metrics and figures."
