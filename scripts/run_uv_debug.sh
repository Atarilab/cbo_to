#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env_var_jax.sh"
# Usage:
#   scripts/run_uv_debug.sh path/to/script.py [args...]
#   scripts/run_uv_debug.sh --module package.module [args...]
#
# Examples:
#   scripts/run_uv_debug.sh your_script.py --foo bar
#   scripts/run_uv_debug.sh --module package.module --foo bar
#   STRACE=1 scripts/run_uv_debug.sh your_script.py
#   LOGDIR=./logs scripts/run_uv_debug.sh your_script.py

SCRIPT="${1:-}"
shift || true

if [[ -z "${SCRIPT}" ]]; then
  echo "Usage: $0 path/to/script.py [args...] | $0 --module package.module [args...]" >&2
  exit 2
fi

RUN_AS_MODULE=0
MODULE=""
if [[ "$SCRIPT" == "-m" || "$SCRIPT" == "--module" ]]; then
  RUN_AS_MODULE=1
  MODULE="${1:-}"
  shift || true
  if [[ -z "$MODULE" ]]; then
    echo "Usage: $0 --module package.module [args...]" >&2
    exit 2
  fi
fi

LOGDIR="${LOGDIR:-./debug-logs}"
mkdir -p "$LOGDIR"

ts="$(date +'%Y%m%d-%H%M%S')"
out="$LOGDIR/run-$ts.out.log"
err="$LOGDIR/run-$ts.err.log"
meta="$LOGDIR/run-$ts.meta.log"

# Environment for better diagnostics
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# Optional: dump stack traces on demand with:
#   kill -USR1 <pid>   (Linux/macOS)
export PYTHONBREAKPOINT=0

# If you want to force CPU-only quickly for debugging GPU issues:
# export CUDA_VISIBLE_DEVICES=""

echo "[meta] started: $(date -Is)" | tee -a "$meta" >/dev/null
if [[ "$RUN_AS_MODULE" -eq 1 ]]; then
  echo "[meta] module: $MODULE $*" | tee -a "$meta" >/dev/null
else
  echo "[meta] script: $SCRIPT $*" | tee -a "$meta" >/dev/null
fi
echo "[meta] uv: $(command -v uv || echo 'NOT FOUND')" | tee -a "$meta" >/dev/null
echo "[meta] python: $(python -V 2>&1 || true)" | tee -a "$meta" >/dev/null
echo "[meta] uname: $(uname -a)" | tee -a "$meta" >/dev/null

# Helpful cleanup/logging on exit
child_pid=""
on_exit() {
  rc=$?
  echo "[meta] ended: $(date -Is)" | tee -a "$meta" >/dev/null
  echo "[meta] exit_code: $rc" | tee -a "$meta" >/dev/null

  # If killed by signal, bash usually reports 128+signal
  if (( rc >= 128 )); then
    sig=$((rc - 128))
    echo "[meta] likely signal: $sig" | tee -a "$meta" >/dev/null
    if (( sig == 9 )); then
      echo "[meta] SIGKILL often means OOM-killer or external kill." | tee -a "$meta" >/dev/null
    fi
  fi

  # Best-effort Linux kernel log hints (OOM / GPU driver resets)
  if [[ "$(uname -s)" == "Linux" ]]; then
    {
      echo ""
      echo "===== kernel hints (last ~5 min) ====="
      # OOM killer
      dmesg -T 2>/dev/null | tail -n 300 | grep -E "Out of memory|Killed process|oom-killer" || true
      # NVIDIA driver resets / Xid errors (if relevant)
      dmesg -T 2>/dev/null | tail -n 300 | grep -E "NVRM|Xid|GPU has fallen off the bus|amdgpu|i915|drm" || true
    } >> "$meta" || true
  fi

  echo "[meta] logs:"
  echo "  stdout: $out"
  echo "  stderr: $err"
  echo "  meta:   $meta"
}
trap on_exit EXIT

# Also react to signals and try to forward them to child
forward_signal() {
  sig="$1"
  echo "[meta] runner got signal $sig, forwarding to child pid=$child_pid" | tee -a "$meta" >/dev/null
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill "-$sig" "$child_pid" 2>/dev/null || true
  fi
}
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM
trap 'forward_signal HUP' HUP

# Build command
if [[ "$RUN_AS_MODULE" -eq 1 ]]; then
  cmd=(uv run python -u -X faulthandler -m "$MODULE" "$@")
else
  cmd=(uv run python -u -X faulthandler "$SCRIPT" "$@")
fi

echo "[meta] command: ${cmd[*]}" | tee -a "$meta" >/dev/null

# Optional deep tracing
if [[ "${STRACE:-0}" == "1" ]] && command -v strace >/dev/null 2>&1; then
  trace="$LOGDIR/run-$ts.strace.log"
  echo "[meta] strace enabled: $trace" | tee -a "$meta" >/dev/null
  cmd=(strace -ff -o "$trace" "${cmd[@]}")
fi

# Run, tee logs, preserve real exit code of python (PIPESTATUS)
set +e
("${cmd[@]}" 1> >(tee -a "$out") 2> >(tee -a "$err" >&2)) &
child_pid=$!
wait "$child_pid"
rc=$?
set -e

exit "$rc"

echo "JAX/XLA environment variables set:"
echo "  XLA_CLIENT_MEM_FRACTION=$XLA_CLIENT_MEM_FRACTION"
echo "  XLA_FLAGS=$XLA_FLAGS"
