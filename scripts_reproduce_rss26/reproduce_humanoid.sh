#!/usr/bin/env bash
# Reproduce the humanoid (G1) benchmark figure (fig:benchmark_best_loss_humanoid_10k).
#
# Paper figure: figs/exp/g1/compare_best_cost_10k.pdf
# Config:       config/rss26_humanoid_decay10_n10k_nested.json
# Seeds:        0, 1, 2
#
# Requirements: >= 32 GB GPU VRAM for N=10000 particles.
#               On a 24 GB machine, use --low-vram to reduce to N=5000.
#
# Usage:
#   scripts_reproduce_rss26/reproduce_humanoid.sh               # run all 3 seeds, N=10000
#   scripts_reproduce_rss26/reproduce_humanoid.sh --seeds 0     # single seed
#   scripts_reproduce_rss26/reproduce_humanoid.sh --low-vram    # reduce to N=5000 for 24G GPU
#
# This wrapper owns reproduction-specific post-processing:
#   1. call scripts/run_compare_seeds.sh for the hardcoded humanoid config
#   2. aggregate only the compare directories created by this invocation
#   3. generate mean/std seed plots and print their location
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="config/rss26_humanoid_decay10_n10k_nested.json"
SEEDS="0,1,2"
LOW_VRAM=0
COMPARE_DIR_MANIFEST="$(mktemp /tmp/humanoid_compare_dirs_XXXXXX.txt)"
trap 'rm -f "$COMPARE_DIR_MANIFEST"' EXIT

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)    SEEDS="$2"; shift 2 ;;
    --low-vram) LOW_VRAM=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Check available GPU VRAM
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)"
VRAM_GB=$(( VRAM_MB / 1024 ))

echo "============================================================"
echo "Humanoid (G1) Reproduction"
echo "  Config:  $CONFIG"
echo "  Seeds:   $SEEDS"
echo "  GPU VRAM detected: ${VRAM_GB} GB"
echo "============================================================"

# Warn if VRAM is too low and --low-vram not specified
if [[ $VRAM_GB -lt 30 && $LOW_VRAM -eq 0 ]]; then
  echo ""
  echo "WARNING: GPU has ${VRAM_GB} GB VRAM. The original run used 10,000 particles"
  echo "         which requires ~32 GB. This may OOM."
  echo ""
  echo "Options:"
  echo "  1) Run with --low-vram flag to use N=5000 instead (fits in 24 GB)."
  echo "     Results will differ slightly from the paper figure."
  echo "  2) Continue anyway (may crash with OOM)."
  echo ""
  read -r -p "Continue with N=10000? [y/N] " confirm
  if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborting. Re-run with --low-vram to use N=5000."
    exit 1
  fi
fi

# Low-VRAM fallback: use a 5k-sample config instead
if [[ $LOW_VRAM -eq 1 ]]; then
  CONFIG_5K="config/gpu1k/cbo_default_base_only_1k_temp_minus_16_decay10.json"
  if [[ -f "$CONFIG_5K" ]]; then
    echo "Low-VRAM mode: switching to $CONFIG_5K (N=1000)"
    CONFIG="$CONFIG_5K"
  else
    # Create a temporary override using python to patch the num_samples field
    TMP_CONFIG="$(mktemp /tmp/humanoid_5k_XXXX.json)"
    python3 -c "
import json, sys
d = json.load(open('$CONFIG'))
d['num_samples'] = 5000
json.dump(d, open('$TMP_CONFIG', 'w'), indent=2)
print('Patched config written to $TMP_CONFIG', file=sys.stderr)
"
    CONFIG="$TMP_CONFIG"
    echo "Low-VRAM mode: patched config with N=5000 at $CONFIG"
  fi
fi

# Run seeds via the standard multi-seed script.
scripts/run_compare_seeds.sh --manifest "$COMPARE_DIR_MANIFEST" "$CONFIG" "$SEEDS"

COMPARE_DIRS=()
while IFS= read -r compare_dir; do
  if [[ -n "$compare_dir" && -d "$compare_dir" ]]; then
    COMPARE_DIRS+=("$compare_dir")
  fi
done < "$COMPARE_DIR_MANIFEST"

if [[ "${#COMPARE_DIRS[@]}" -eq 0 ]]; then
  echo "No compare directories were created; skipping aggregation." >&2
  exit 0
fi

AGG_DIR="tmp/agg_humanoid_$(date +%Y%m%d_%H%M%S)"
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
echo ""
echo "============================================================"
