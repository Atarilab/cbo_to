#!/usr/bin/env bash
# Reproduce the double cartpole benchmark figure (fig:benchmark_dCartpole).
#
# Config:       config/rss26_dcartpole_decay10_n5k_nested.json
# Seeds:        0, 1, 2  (~55 min each on RTX 5090)
#
# Usage:
#   scripts_reproduce_rss26/reproduce_double_cartpole.sh              # run all 3 seeds
#   scripts_reproduce_rss26/reproduce_double_cartpole.sh --seeds 0    # run one seed
#   scripts_reproduce_rss26/reproduce_double_cartpole.sh --debug      # all seeds, few iterations, full aggregation path
#   scripts_reproduce_rss26/reproduce_double_cartpole.sh --debug --seeds 0  # fastest single-seed smoke
#
# This wrapper owns reproduction-specific post-processing:
#   1. call scripts/run_compare_seeds.sh for the hardcoded double-cartpole config
#   2. aggregate only the compare directories created by this invocation
#   3. generate mean/std seed plots and print their location
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="config/rss26_dcartpole_decay10_n5k_nested.json"
SEEDS="0,1,2"
DEBUG=0
DEBUG_MAX_ITER=3
COMPARE_ARGS=()
COMPARE_DIR_MANIFEST="$(mktemp /tmp/dcartpole_compare_dirs_XXXXXX.txt)"
DEBUG_CONFIG=""

cleanup() {
  rm -f "$COMPARE_DIR_MANIFEST"
  if [[ -n "$DEBUG_CONFIG" ]]; then
    rm -f "$DEBUG_CONFIG"
  fi
}
trap cleanup EXIT

# Parse optional arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2 ;;
    --debug) DEBUG=1; shift ;;
    --debug-max-iter) DEBUG_MAX_ITER="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$DEBUG" -eq 1 ]]; then
  DEBUG_CONFIG="$(mktemp /tmp/dcartpole_smoke_config_XXXXXX.json)"
  .venv/bin/python - "$CONFIG" "$DEBUG_CONFIG" "$DEBUG_MAX_ITER" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
max_iter = int(sys.argv[3])

data = json.loads(source.read_text(encoding="utf-8"))
if "run" in data or "algorithm_defaults" in data:
    data.setdefault("run", {})["max_iter"] = max_iter
    defaults = data.setdefault("algorithm_defaults", {})
    defaults["num_samples"] = 10
    defaults["persist_particles_every"] = None
    defaults["persist_particles_latest"] = None
    defaults["persist_particle_gif_count"] = 0
    logging = data.setdefault("logging", {})
    logging["mean_knots_cost_every"] = 1
else:
    data["max_iter"] = max_iter
    data["num_samples"] = 10
    data["persist_particles_every"] = None
    data["persist_particles_latest"] = None
    data["persist_particle_gif_count"] = 0
    data["mean_knots_cost_every"] = 1

target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
  CONFIG="$DEBUG_CONFIG"
fi

echo "============================================================"
echo "Double Cartpole Reproduction"
echo "  Config:  $CONFIG"
echo "  Seeds:   $SEEDS"
if [[ "$DEBUG" -eq 1 ]]; then
  echo "  Mode:    debug smoke test (num_samples=10, max_iter=$DEBUG_MAX_ITER, no GIFs)"
fi
echo "============================================================"

COMPARE_DIRS=()
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

echo ""
echo "==> Running config: $CONFIG"
: > "$COMPARE_DIR_MANIFEST"
scripts/run_compare_seeds.sh --manifest "$COMPARE_DIR_MANIFEST" "$CONFIG" "$SEEDS" "${COMPARE_ARGS[@]}"

while IFS= read -r compare_dir; do
  if [[ -n "$compare_dir" && -d "$compare_dir" ]]; then
    COMPARE_DIRS+=("$compare_dir")
  fi
done < "$COMPARE_DIR_MANIFEST"

if [[ "${#COMPARE_DIRS[@]}" -eq 0 ]]; then
  echo "No compare directories were created; skipping aggregation." >&2
  exit 0
fi

if [[ "$DEBUG" -eq 1 ]]; then
  AGG_DIR="tmp/agg_double_cartpole_smoke_$(date +%Y%m%d_%H%M%S)"
else
  AGG_DIR="tmp/agg_double_cartpole_$(date +%Y%m%d_%H%M%S)"
fi
agg_args=("$AGG_DIR")
for compare_dir in "${COMPARE_DIRS[@]}"; do
  agg_args+=("--compare-dir" "$compare_dir")
done

echo ""
echo "==> Aggregating results from this reproduction run..."
agg_output="$(.venv/bin/python -m runners.aggregate_compare_costs "${agg_args[@]}")"
echo "$agg_output"

echo "==> Plotting aggregated seed curves from $AGG_DIR"
while IFS= read -r task_particles_dir; do
  .venv/bin/python -m plots.cost_folder_seeds "$task_particles_dir"
done < <(find "$AGG_DIR" -mindepth 2 -maxdepth 2 -type d | sort)

echo ""
echo "============================================================"
echo "Reproduction complete."
echo "Aggregated output:"
echo "  $AGG_DIR"
echo ""
echo "Aggregated plots:"
find "$AGG_DIR" -name "*.pdf" | sort | sed 's/^/  /'
echo "============================================================"
