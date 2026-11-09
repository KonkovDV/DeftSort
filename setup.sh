#!/bin/bash
# Optional Rust acceleration (team048_accel). Runs once at install time (internet available); NEVER fails the install:
# without the extension the solution uses its NumPy reference implementation (identical results, slower planning).
# 1) vendored prebuilt wheel: manylinux2014 x86_64, abi3 (CPython >= 3.10), baseline x86-64 (no AVX), glibc >= 2.14
# 2) fallback: build from rust/team048_accel with rustup + maturin (offline run.sh never needs the network)
cd "$(dirname "$0")" || exit 0
PY=${TEAM048_PY:-/usr/bin/python3}
ok() { "$PY" -c "import team048_accel" >/dev/null 2>&1; }
log() { echo "[setup.sh] $*"; }
WHL=$(ls vendor/team048_accel-*-abi3-manylinux*.whl 2>/dev/null | head -1)
if [ "${TEAM048_FORCE_BUILD:-0}" != "1" ] && [ -n "$WHL" ] && (cd vendor && sha256sum -c --quiet SHA256SUMS >/dev/null 2>&1); then
  timeout 300 "$PY" -m pip install --no-deps --no-index --force-reinstall "$WHL" >/dev/null 2>&1 || log "vendored wheel did not install"
fi
if ! ok && [ "${TEAM048_NO_BUILD:-0}" != "1" ]; then
  log "building team048_accel from source"
  export CARGO_HOME=${CARGO_HOME:-/opt/team048/cargo} RUSTUP_HOME=${RUSTUP_HOME:-/opt/team048/rustup}
  export PATH="$CARGO_HOME/bin:$PATH"
  if ! command -v cargo >/dev/null 2>&1; then
    timeout 600 bash -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path' >/dev/null 2>&1 || log "rustup failed"
  fi
  if ! command -v cc >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then   # rustc needs a system linker
    timeout 600 bash -c 'apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc libc6-dev' >/dev/null 2>&1 || log "no C linker"
  fi
  if command -v cargo >/dev/null 2>&1; then
    T=$(mktemp -d)
    timeout 300 "$PY" -m pip install --target "$T/m" "maturin>=1.7,<2" >/dev/null 2>&1 &&
      (cd rust/team048_accel && PYTHONPATH="$T/m" timeout 900 "$T/m/bin/maturin" build --release -i "$PY" -o "$T/w" >/dev/null 2>&1) &&
      timeout 300 "$PY" -m pip install --no-deps --force-reinstall "$T"/w/team048_accel-*.whl >/dev/null 2>&1 || log "source build failed"
    rm -rf "$T"
  fi
fi
if ok; then log "team048_accel $("$PY" -c 'import team048_accel as m;print(m.__version__)') ready"; else log "team048_accel unavailable -> NumPy reference"; fi
exit 0
