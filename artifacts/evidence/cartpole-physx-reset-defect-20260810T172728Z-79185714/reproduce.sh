#!/usr/bin/env bash
# Reproduce IVF run cartpole-physx-reset-defect-20260810T172728Z-79185714
#
# This script re-executes the manifest that produced this evidence bundle. It does not
# guarantee identical output on a different machine: identical output is a property of
# the workload, not of this script. `ivf reproduce` compares the two and reports what
# changed.
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
ivf --results-root ivf-results validate "${repository_root}/validation/examples/cartpole_reset_defect.yaml"
