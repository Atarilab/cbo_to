from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from runners.particle_replay import load_costs, resolve_experiment_dir, resolve_particles_path


def _infer_iteration_tag(particles_path: Path) -> str:
    stem = particles_path.stem
    if stem.startswith("particles_iter_"):
        return f"iter_{stem.split('_')[-1]}"
    latest_iter_path = particles_path.parent / "particles_latest_iter.txt"
    if latest_iter_path.exists():
        try:
            iter_val = int(latest_iter_path.read_text(encoding="utf-8").strip())
            return f"iter_{iter_val:05d}"
        except ValueError:
            pass
    return "iter_unknown"


def _load_particles_and_costs(
    experiment_dir: str | Path | None,
    *,
    algorithm: str | None,
    iteration: int | None,
) -> tuple[Path, np.ndarray, np.ndarray, str]:
    which = "iter" if iteration is not None else "latest"
    source = resolve_experiment_dir(experiment_dir, algorithm=algorithm)
    particles_path = resolve_particles_path(source, which=which, iteration=iteration)
    particles = np.load(particles_path)
    try:
        costs = load_costs(source, which=which, iteration=iteration)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Costs file not found for {particles_path}; re-run with costs saved first."
        ) from exc
    iter_tag = _infer_iteration_tag(particles_path)
    return source.experiment_dir, particles, costs, iter_tag


def _select_top_indices(costs: np.ndarray, count: int) -> np.ndarray:
    costs = np.asarray(costs).reshape(-1)
    order = np.argsort(costs)
    count = max(1, min(int(count), order.shape[0]))
    return order[:count]


def _plot_knots(
    particles: np.ndarray,
    costs: np.ndarray,
    indices: np.ndarray,
    *,
    out_dir: Path,
    cols: int,
    show: bool,
    iter_tag: str,
) -> None:
    if particles.ndim != 3:
        raise ValueError(f"Expected particles with shape (N,K,nu), got {particles.shape}")
    num_knots = particles.shape[1]
    nu = particles.shape[2]

    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = max(1, min(int(cols), nu))
    rows = int(np.ceil(nu / cols))
    x = np.arange(num_knots + 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, idx in enumerate(indices):
        knots = particles[int(idx)]
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.2 * rows), sharex=True)
        axes = np.asarray(axes).reshape(-1)
        for dim in range(nu):
            ax = axes[dim]
            y = np.append(knots[:, dim], knots[-1, dim])
            ax.step(x, y, where="post", linewidth=1.2)
            ax.axhline(0.0, linestyle="--", linewidth=0.8, color="0.5")
            abs_max = float(np.max(np.abs(y)))
            if abs_max > 0.0:
                pad = max(0.1 * abs_max, 1e-6)
                ax.set_ylim(-abs_max - pad, abs_max + pad)
                ax.set_yticks([-abs_max, 0.0, abs_max])
            ax.set_ylabel(f"u[{dim}]")
        for dim in range(nu, len(axes)):
            axes[dim].axis("off")
        cost = float(costs[int(idx)])
        fig.suptitle(f"rank {rank} idx {idx} cost {cost:.6g}")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_path = (
            out_dir
            / f"knots_{iter_tag}_rank_{rank:04d}_idx_{idx:04d}_cost_{cost:.6g}.pdf"
        )
        fig.savefig(out_path)
        if show:
            plt.show()
        plt.close(fig)
        print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot top-ranked knot sequences from saved particles."
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Experiment directory (defaults to latest under tmp/experiments).",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Algorithm subdir when using compare runs (e.g. CBOx).",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Iteration to load (default: latest particles).",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=5,
        help="Number of top-ranked particles to plot.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=2,
        help="Number of subplot columns for control dimensions.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <experiment_dir>/plot_knots).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (also saves PNGs).",
    )
    args = parser.parse_args()

    experiment_dir, particles, costs, iter_tag = _load_particles_and_costs(
        args.experiment_dir,
        algorithm=args.algorithm,
        iteration=args.iteration,
    )
    indices = _select_top_indices(costs, args.count)

    out_dir = Path(args.out_dir) if args.out_dir else (Path(experiment_dir) / "plot_knots")
    _plot_knots(
        particles,
        costs,
        indices,
        out_dir=out_dir,
        cols=args.cols,
        show=args.show,
        iter_tag=iter_tag,
    )


if __name__ == "__main__":
    main()
