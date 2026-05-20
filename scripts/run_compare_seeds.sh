#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF' >&2
Usage: scripts/run_compare_seeds.sh [options] <config.json> <seed1,seed2,...> [compare args...]

Options:
  --manifest path              Write one created compare directory per line.
  --after-seed-callback path   Executable called after each seed run as:
                                 path <compare_dir> <seed> <config_path>

Examples:
  scripts/run_compare_seeds.sh config/gpu1k/cbo_default_base_only_1k_temp_minus_16_decay10.json seed1,seed2,seed3
  scripts/run_compare_seeds.sh config/gpu32/cbo_default_base_only_temp_minus_16_decay10.json 1,2,3 --algo CBOx,CMA-ES
  scripts/run_compare_seeds.sh --manifest /tmp/compare_dirs.txt config/dCartpole/cbo_default_base_only_temp_minus_16_decay10_samplesize5k.json 0,1,2
  scripts/run_compare_seeds.sh --after-seed-callback scripts/plot_pcbo_multimodality.sh config/dCartpole/sves_pcbo_nested_config.json 0,1,2
EOF
}

manifest_path=""
after_seed_callback=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      if [[ $# -lt 2 ]]; then
        echo "--manifest requires a path." >&2
        usage
        exit 2
      fi
      manifest_path="$2"
      shift 2
      ;;
    --after-seed-callback)
      if [[ $# -lt 2 ]]; then
        echo "--after-seed-callback requires a path." >&2
        usage
        exit 2
      fi
      after_seed_callback="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

config_path="$1"
seed_list="$2"
shift 2
extra_args=("$@")

if [[ -n "$after_seed_callback" && ! -x "$after_seed_callback" ]]; then
  echo "Callback is not executable: $after_seed_callback" >&2
  exit 2
fi

IFS=',' read -r -a seeds <<< "$seed_list"
if [[ "${#seeds[@]}" -eq 0 ]]; then
  echo "No seeds provided." >&2
  usage
  exit 2
fi

latest_compare_dir() {
  find "tmp/experiments" -maxdepth 1 -type d -name 'compare_*' | sort | tail -n 1
}

created_compare_dirs=()
if [[ -n "$manifest_path" ]]; then
  : > "$manifest_path"
fi

for raw_seed in "${seeds[@]}"; do
  seed="$(echo "$raw_seed" | tr -d '[:space:]')"
  seed="${seed#seed}"
  if [[ -z "$seed" ]]; then
    echo "Empty seed in list: '$raw_seed'" >&2
    exit 2
  fi
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed '$raw_seed' (expected digits or 'seedN')." >&2
    exit 2
  fi

  echo "==> Running runners.compare with --seed $seed"
  scripts/run_uv_debug.sh --module runners.compare "$config_path" --seed "$seed" "${extra_args[@]}"

  compare_dir="$(latest_compare_dir)"
  if [[ -n "$compare_dir" && -d "$compare_dir" ]]; then
    created_compare_dirs+=("$compare_dir")
    if [[ -n "$manifest_path" ]]; then
      echo "$compare_dir" >> "$manifest_path"
    fi
    if [[ -n "$after_seed_callback" ]]; then
      "$after_seed_callback" "$compare_dir" "$seed" "$config_path"
    fi
  fi
done

if [[ "${#created_compare_dirs[@]}" -eq 0 ]]; then
  echo "No compare directories were created." >&2
  exit 0
fi

declare -A seen_compare_dirs=()
echo ""
echo "Compare directories created:"
for compare_dir in "${created_compare_dirs[@]}"; do
  if [[ -n "${seen_compare_dirs[$compare_dir]:-}" ]]; then
    continue
  fi
  seen_compare_dirs["$compare_dir"]=1
  echo "  $compare_dir"
done
