"""Shared plotting conventions for benchmark figures."""

from __future__ import annotations


ALGORITHM_COLORS = {
    "CBO": "#1b9e77",
    "CBOx": "#1b9e77",
    "CBOxPolarized": "#1b9e77",
    "PolarizedCBO": "#1b9e77",
    "CMA": "#d95f02",
    "CMA-ES": "#d95f02",
    "SV-CMA-ES": "#d95f02",
    "MPPI": "#7570b3",
    "MPPI_lr": "#7570b3",
    "MPPI-lr": "#7570b3",
}

FALLBACK_ALGORITHM_COLORS = [
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]

ALGORITHM_LABELS = {
    "CBOx": "CBO",
    "CBOxPolarized": "CBO",
    "PolarizedCBO": "CBO",
    "MPPI_lr": "MPPI",
    "MPPI-lr": "MPPI",
    "CMA": "CMA-ES",
}


def display_algo_name(algo_name: str) -> str:
    """Return the label used in paper figures for an algorithm id."""
    return ALGORITHM_LABELS.get(algo_name, algo_name)


def algorithm_color(algo_name: str) -> str:
    """Return a stable paper color for an algorithm id."""
    if algo_name in ALGORITHM_COLORS:
        return ALGORITHM_COLORS[algo_name]

    # Deterministic fallback for algorithms outside the paper comparison set.
    idx = sum((i + 1) * ord(ch) for i, ch in enumerate(algo_name))
    return FALLBACK_ALGORITHM_COLORS[idx % len(FALLBACK_ALGORITHM_COLORS)]
