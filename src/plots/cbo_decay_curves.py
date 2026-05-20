#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DECAY_PATTERN = re.compile(r"_cbo_decay_([^/]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="cost decay"
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("tmp/experiments"),
        help="Directory containing compare_*_cbo_decay_* folders.",
    )
    parser.add_argument(
        "--algo",
        default="CBOx",
        help="Algorithm prefix used in <algo>_cost_per_iter.npy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to <root>/<algo>_cbo_decay_compare.pdf.",
    )
    return parser.parse_args()


def extract_decay(path: Path) -> str | None:
    match = DECAY_PATTERN.search(path.name)
    if match is None:
        return None
    return match.group(1)


def decay_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def main() -> None:
    args = parse_args()
    curves: list[tuple[str, np.ndarray]] = []

    for folder in sorted(args.root.glob("compare*_cbo_decay_*")):
        if not folder.is_dir():
            continue
        decay = extract_decay(folder)
        if decay is None:
            continue
        npy_path = folder / f"{args.algo}_cost_per_iter.npy"
        if not npy_path.exists():
            continue
        curves.append((decay, np.load(npy_path)))

    if not curves:
        raise FileNotFoundError(
            f"No {args.algo}_cost_per_iter.npy files found in {args.root}"
        )

    curves.sort(key=lambda item: decay_sort_key(item[0]))

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for decay, costs in curves:
        ax.plot(np.asarray(costs), label=rf"$\lambda={decay}$")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.grid(True)
    ax.legend()

    output_path = args.output
    if output_path is None:
        output_path = args.root / f"{args.algo}_cbo_decay_compare.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
