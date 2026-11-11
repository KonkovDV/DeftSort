#!/usr/bin/env bash
# One-command offline check: tests with the Rust module (if installed) and with the NumPy reference only,
# then the Rust crate tests if cargo is available. Usage: bash tools/verify.sh   (from any directory)
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
run() { "$@" 2>&1 | tail -n 1; }
accel=$("$PY" -c 'import sys; sys.path.insert(0, "src/team048_solution"); from team048_solution import _accel; print("Rust API %d" % _accel.API if _accel.enabled() else "not installed")' 2>/dev/null || echo "unknown")
echo "team048_accel: $accel"
echo "pytest, Rust if available : $(run "$PY" -m pytest -q -p no:cacheprovider)"
echo "pytest, TEAM048_ACCEL=0   : $(TEAM048_ACCEL=0 run "$PY" -m pytest -q -p no:cacheprovider)"
if command -v cargo >/dev/null 2>&1; then
  echo "cargo test --release      : $(cd rust/team048_accel && cargo test --release 2>&1 | grep -m1 '^test result')"
else
  echo "cargo test --release      : skipped (cargo not found)"
fi
