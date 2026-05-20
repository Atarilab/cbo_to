#!/usr/bin/env bash

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# =========================
# JAX / XLA GPU MEMORY
# =========================

# Give CUDA driver headroom for CUDA graph / command buffer instantiation
export XLA_CLIENT_MEM_FRACTION=0.75

# Disable CUDA command buffers (fixes CUDA graph OOM)
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

# =========================
# OPTIONAL: DEBUG / SAFETY
# =========================

# Prevent JAX from silently allocating too aggressively
export JAX_ENABLE_X64=0

# Make async errors easier to debug (optional, slower)
# export CUDA_LAUNCH_BLOCKING=1

# Log XLA memory behavior (optional)
# export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_autotune_level=2"

echo "JAX/XLA environment variables set:"
echo "  XLA_CLIENT_MEM_FRACTION=$XLA_CLIENT_MEM_FRACTION"
echo "  XLA_FLAGS=$XLA_FLAGS"
