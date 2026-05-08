#!/usr/bin/env bash
# Z_run_all.sh – legacy compatibility wrapper.
#
# The canonical entry-point is the `cdriver-run` console script installed by
# `pip install -e .`.  This script exists solely for backwards compatibility
# with workflows that called the Python script directly.
#
# Usage (from repo root, after `pip install -e .`):
#   bash scripts_and_results/Z_run_all.sh [extra args passed to cdriver-run]

set -euo pipefail

echo "[Z_run_all.sh] Delegating to cdriver-run …"
cdriver-run "$@"
