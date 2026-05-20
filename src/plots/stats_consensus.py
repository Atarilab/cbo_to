from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = []
        for row in reader:
            if not row:
                continue
            rows.append([float(item) for item in row])
    return header, rows


def _is_cbox_dir(path: Path) -> bool:
    for part in path.parts:
        lower = part.lower()
        if "cbox" in lower or "cbo_x" in lower:
            return True
    return False


def _ensure_cbox_dir(path: Path) -> None:
    if not _is_cbox_dir(path):
        raise ValueError(f"Expected a CBOx experiment dir, got: {path}")


def plot_loss_curve(stats_dir: str | Path) -> None:
    stats_dir = Path(stats_dir)
    csv_path = stats_dir / "loss_curve.csv"
    header, rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No data in {csv_path}")
    columns = list(zip(*rows))
    col_map = {name: columns[i] for i, name in enumerate(header)}
    iteration = col_map.get("iteration")
    best = col_map.get("best_cost")
    mean = col_map.get("mean_cost")
    if iteration is None or best is None or mean is None:
        raise ValueError(f"Missing expected columns in {csv_path}: {header}")
    fig, ax = plt.subplots()
    ax.plot(iteration, best, label="best_cost")
    ax.plot(iteration, mean, label="mean_cost")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_consensus_stats(stats_dir: str | Path) -> None:
    stats_dir = Path(stats_dir)
    csv_path = stats_dir / "consensus_stats.csv"
    header, rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No data in {csv_path}")
    columns = list(zip(*rows))
    col_map = {name: columns[i] for i, name in enumerate(header)}
    iteration = col_map.pop("iteration", None)
    if iteration is None:
        raise ValueError(f"Missing iteration column in {csv_path}: {header}")
    fig, ax = plt.subplots()
    last_values = {}
    for name, values in col_map.items():
        ax.plot(iteration, values, label=name)
        last_values[name] = values[-1]
    _add_right_margin_labels(ax, last_values)
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def _find_latest_experiment_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"Experiments dir not found: {base_dir}")
    candidates = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No experiments found under {base_dir}")
    for candidate in reversed(candidates):
        if candidate.name.startswith("compare_"):
            subdirs = sorted([p for p in candidate.iterdir() if p.is_dir()])
            for subdir in reversed(subdirs):
                if (subdir / "stats").exists() and _is_cbox_dir(subdir):
                    return subdir
            continue
        if (candidate / "stats").exists() and _is_cbox_dir(candidate):
            return candidate
    raise FileNotFoundError(f"No CBOx experiments found under {base_dir}")


def _find_stats_dir(experiment_dir: Path) -> Path:
    stats_dir = experiment_dir / "stats"
    if stats_dir.exists():
        return stats_dir
    candidates = sorted([p for p in experiment_dir.iterdir() if p.is_dir()])
    for candidate in reversed(candidates):
        candidate_stats = candidate / "stats"
        if candidate_stats.exists():
            return candidate_stats
    raise FileNotFoundError(f"No stats dir found under {experiment_dir}")


def _save_loss_curve_pdf(stats_dir: Path) -> Path:
    csv_path = stats_dir / "loss_curve.csv"
    header, rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No data in {csv_path}")
    columns = list(zip(*rows))
    col_map = {name: columns[i] for i, name in enumerate(header)}
    iteration = col_map.get("iteration")
    best = col_map.get("best_cost")
    mean = col_map.get("mean_cost")
    if iteration is None or best is None or mean is None:
        raise ValueError(f"Missing expected columns in {csv_path}: {header}")
    fig, ax = plt.subplots()
    ax.plot(iteration, best, label="best_cost")
    ax.plot(iteration, mean, label="mean_cost")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    out_dir = stats_dir / "stats_figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loss_curve.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_best_loss_pdf(stats_dir: Path) -> Path:
    csv_path = stats_dir / "loss_curve.csv"
    header, rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No data in {csv_path}")
    columns = list(zip(*rows))
    col_map = {name: columns[i] for i, name in enumerate(header)}
    iteration = col_map.get("iteration")
    best = col_map.get("best_cost")
    if iteration is None or best is None:
        raise ValueError(f"Missing expected columns in {csv_path}: {header}")
    fig, ax = plt.subplots()
    ax.plot(iteration, best, label="best_cost")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    out_dir = stats_dir / "stats_figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_loss.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_mean_loss_pdf(stats_dir: Path) -> Path:
    csv_path = stats_dir / "loss_curve.csv"
    header, rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No data in {csv_path}")
    columns = list(zip(*rows))
    col_map = {name: columns[i] for i, name in enumerate(header)}
    iteration = col_map.get("iteration")
    mean = col_map.get("mean_cost")
    if iteration is None or mean is None:
        raise ValueError(f"Missing expected columns in {csv_path}: {header}")
    fig, ax = plt.subplots()
    ax.plot(iteration, mean, label="mean_cost")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    out_dir = stats_dir / "stats_figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mean_loss.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_consensus_stats_pdf(stats_dir: Path) -> Path:
    csv_path = stats_dir / "consensus_stats.csv"
    header, rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No data in {csv_path}")
    columns = list(zip(*rows))
    col_map = {name: columns[i] for i, name in enumerate(header)}
    iteration = col_map.pop("iteration", None)
    if iteration is None:
        raise ValueError(f"Missing iteration column in {csv_path}: {header}")
    fig, ax = plt.subplots()
    last_values = {}
    for name, values in col_map.items():
        ax.plot(iteration, values, label=name)
        last_values[name] = values[-1]
    _add_right_margin_labels(ax, last_values)
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    out_dir = stats_dir / "stats_figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "consensus_stats.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _add_right_margin_labels(ax, last_values: dict[str, float]) -> None:
    if not last_values:
        return
    sorted_items = sorted(last_values.items(), key=lambda item: item[1])
    count = len(sorted_items)
    y_positions = [0.05 + (0.9 * i / max(count - 1, 1)) for i in range(count)]
    for (name, _), y_pos in zip(sorted_items, y_positions):
        ax.text(
            1.01,
            y_pos,
            name,
            transform=ax.transAxes,
            va="center",
            fontsize=8,
        )


def _save_consensus_delta_pdf(stats_dir: Path) -> Path:
    csv_path = stats_dir / "consensus_stats.csv"
    header, rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No data in {csv_path}")
    columns = list(zip(*rows))
    col_map = {name: columns[i] for i, name in enumerate(header)}
    iteration = col_map.get("iteration")
    delta = col_map.get("consensus_delta")
    to_best = col_map.get("consensus_to_best")
    if iteration is None or delta is None or to_best is None:
        raise ValueError(f"Missing expected columns in {csv_path}: {header}")
    fig, ax = plt.subplots()
    ax.plot(iteration, delta, label="consensus_delta")
    ax.plot(iteration, to_best, label="consensus_to_best")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    out_dir = stats_dir / "stats_figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "consensus_delta_to_best.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_particles_to_consensus_pdf(stats_dir: Path) -> Path:
    csv_path = stats_dir / "consensus_stats.csv"
    header, rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No data in {csv_path}")
    columns = list(zip(*rows))
    col_map = {name: columns[i] for i, name in enumerate(header)}
    iteration = col_map.get("iteration")
    mean = col_map.get("particles_to_consensus_mean")
    std = col_map.get("particles_to_consensus_std")
    if iteration is None or mean is None or std is None:
        raise ValueError(f"Missing expected columns in {csv_path}: {header}")
    fig, ax = plt.subplots()
    ax.plot(iteration, mean, label="particles_to_consensus_mean")
    ax.plot(iteration, std, label="particles_to_consensus_std")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    out_dir = stats_dir / "stats_figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "particles_to_consensus.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CBOx experiment stats.")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Path to a specific experiment directory.",
    )
    args = parser.parse_args()

    base_dir = Path("tmp/experiments")
    experiment_dir = (
        args.experiment_dir
        if args.experiment_dir is not None
        else _find_latest_experiment_dir(base_dir)
    )
    stats_dir = _find_stats_dir(experiment_dir)
    _ensure_cbox_dir(stats_dir.parent)
    print(f"Experiment results dir: {stats_dir.parent}")
    pdf_path = _save_loss_curve_pdf(stats_dir)
    consensus_path = _save_consensus_stats_pdf(stats_dir)
    consensus_delta_path = _save_consensus_delta_pdf(stats_dir)
    particles_to_consensus_path = _save_particles_to_consensus_pdf(stats_dir)
    best_loss_path = _save_best_loss_pdf(stats_dir)
    mean_loss_path = _save_mean_loss_pdf(stats_dir)
    print(f"Saved loss curve to {pdf_path}")
    print(f"Saved consensus stats to {consensus_path}")
    print(f"Saved consensus delta/to_best to {consensus_delta_path}")
    print(f"Saved particles->consensus stats to {particles_to_consensus_path}")
    print(f"Saved best loss to {best_loss_path}")
    print(f"Saved mean loss to {mean_loss_path}")


if __name__ == "__main__":
    main()
