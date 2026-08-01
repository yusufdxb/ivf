#!/usr/bin/env bash
# Reproduce IVF run synthetic-reset-velocity-defect-20260801T050934Z-903ee8a4
#
# This script re-executes the manifest that produced this evidence bundle. It does not
# guarantee identical output on a different machine: identical output is a property of
# the workload, not of this script. `ivf reproduce` compares the two and reports what
# changed.
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
ivf --results-root ivf-results validate "${repository_root}/validation/examples/synthetic_fault.yaml"
