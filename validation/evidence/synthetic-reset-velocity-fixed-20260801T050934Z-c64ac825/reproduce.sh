#!/usr/bin/env bash
# Reproduce IVF run synthetic-reset-velocity-fixed-20260801T050934Z-c64ac825
#
# This script re-executes the manifest that produced this evidence bundle. It does not
# guarantee identical output on a different machine: identical output is a property of
# the workload, not of this script. `ivf reproduce` compares the two and reports what
# changed.
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
ivf --results-root ivf-results validate "${repository_root}/validation/examples/synthetic_fixed.yaml"
