#!/usr/bin/env bash
# One-command offline check: tests with the Rust module (if installed) and with the NumPy reference only, the
# documentation facts, the lint gate, then the Rust crate tests if cargo is available. One result line per step.
# Usage: bash tools/verify.sh   (from any directory; PYTHON=... selects the interpreter)
set -u
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python3}
run() { "$@" 2>&1 | tail -n 1; }
accel=$("$PY" -c 'import sys; sys.path.insert(0, "src/team048_solution"); from team048_solution import _accel; print("Rust API %d" % _accel.API if _accel.enabled() else "not installed")' 2>/dev/null || echo "unknown")
echo "team048_accel: $accel"
echo "pytest, Rust if available : $(run "$PY" -m pytest -q -p no:cacheprovider)"
echo "pytest, TEAM048_ACCEL=0   : $(TEAM048_ACCEL=0 run "$PY" -m pytest -q -p no:cacheprovider)"
echo "docs vs repository        : $(run "$PY" tools/check_docs.py)"
if "$PY" -m ruff --version >/dev/null 2>&1; then
  echo "ruff check                : $(run "$PY" -m ruff check .)"
elif command -v ruff >/dev/null 2>&1; then
  echo "ruff check                : $(run ruff check .)"
else
  echo "ruff check                : skipped (ruff not found; pip install -r requirements-dev.txt)"
fi
if command -v cargo >/dev/null 2>&1; then
  echo "cargo test --release      : $(cd rust/team048_accel && cargo test --release 2>&1 | grep -m1 '^test result')"
else
  echo "cargo test --release      : skipped (cargo not found)"
fi
