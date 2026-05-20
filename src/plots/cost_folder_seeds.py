import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

from plots.style import algorithm_color, display_algo_name


def parse_filename(path: Path) -> Tuple[str, str, str, str | None]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Expected filename like <algo>_<num_samples>_<temperature>.csv, got {path.name}"
        )
    num_index = None
    for idx, part in enumerate(parts):
        if part.isdigit():
            num_index = idx
            break
    if num_index is None or num_index == 0 or num_index + 1 >= len(parts):
        raise ValueError(
            f"Expected filename like <algo>_<num_samples>_<temperature>.csv, got {path.name}"
        )
    algo_name = "_".join(parts[:num_index])
    num_samples = parts[num_index]
    temperature = parts[num_index + 1]
    seed = None
    if len(parts) > num_index + 2:
        seed_candidate = parts[num_index + 2]
        if seed_candidate.startswith("seed") and seed_candidate[4:].isdigit():
            seed = seed_candidate
    return algo_name, num_samples, temperature, seed


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_columns_file(path: Path) -> List[Tuple[str, List[str]]]:
    specs: List[Tuple[str, List[str]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if not parts:
                continue
            label = parts[0]
            candidates = parts[1:] if len(parts) > 1 else [label]
            specs.append((label, candidates))
    return specs


def resolve_column_name(fieldnames: Iterable[str], candidates: List[str]) -> str | None:
    field_set = {name for name in fieldnames}
    for candidate in candidates:
        if candidate in field_set:
            return candidate
    return None


def sanitize_filename(value: str) -> str:
    safe = value.strip().replace(" ", "_")
    safe = safe.replace("/", "_").replace("\\", "_")
    return safe or "plot"


def _parse_iteration(value: str | None) -> int:
    try:
        return int(float(value or "0"))
    except ValueError:
        return 0


def _parse_value(value: str | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _series_by_iteration(rows: List[Dict[str, str]], column_name: str) -> Dict[int, float]:
    series: Dict[int, float] = {}
    for row in rows:
        iteration = _parse_iteration(row.get("iteration"))
        series[iteration] = _parse_value(row.get(column_name))
    return series


def _aggregate_series(series_list: List[Dict[int, float]]) -> Tuple[List[int], List[float], List[float]]:
    iter_to_values: Dict[int, List[float]] = {}
    for series in series_list:
        for iteration, value in series.items():
            if not math.isfinite(value):
                continue
            iter_to_values.setdefault(iteration, []).append(value)
    iterations = sorted(iter_to_values.keys())
    means: List[float] = []
    stds: List[float] = []
    for iteration in iterations:
        values = iter_to_values[iteration]
        if not values:
            continue
        means.append(statistics.fmean(values))
        stds.append(statistics.pstdev(values) if len(values) > 1 else 0.0)
    return iterations, means, stds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot costs from CSV files named <algo>_<num_samples>_<temperature>.csv "
            "with optional _seedX suffix."
        )
    )
    parser.add_argument("folder", type=Path, help="Folder containing CSV files")
    parser.add_argument(
        "--columns-file",
        type=Path,
        help=(
            "Optional txt file with lines like: plot_label,col_name1,col_name2,..."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output image path (e.g., plot.pdf).",
    )
    parser.add_argument(
        "--no-log-scale",
        action="store_true",
        help="Disable log scale on the y-axis.",
    )
    parser.add_argument(
        "--include-temperature",
        action="store_true",
        help="Include temperature in the legend labels.",
    )
    parser.add_argument(
        "--population-size",
        action="store_true",
        help="Include population size in the legend labels.",
    )
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    csv_paths = sorted(folder.glob("*.csv"))
    if not csv_paths:
        nested_csvs: List[Path] = []
        for child in sorted(p for p in folder.iterdir() if p.is_dir()):
            stats_dir = child / "stats"
            if not stats_dir.exists():
                continue
            for candidate in sorted(stats_dir.glob("*.csv")):
                if candidate.name.startswith(f"{child.name}_"):
                    nested_csvs.append(candidate)
        csv_paths = nested_csvs
    if not csv_paths:
        raise SystemExit(f"No CSV files found under {folder}")

    if args.columns_file:
        column_specs = parse_columns_file(args.columns_file)
        if not column_specs:
            raise SystemExit(f"No valid columns in {args.columns_file}")
    else:
        column_specs = [
            ("mean_knots_cost", ["mean_knots_cost"]),
            ("best_cost", ["best_cost"]),
            ("best_ever_cost", ["best_ever_cost"]),
            ("mean_cost", ["mean_cost"]),
        ]

    algo_markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    temp_linestyles = ["-", "--", ":", "-."]
    algo_marker_map: Dict[str, str] = {}
    temp_style_map: Dict[str, str] = {}

    grouped_paths: Dict[Tuple[str, str, str], List[Path]] = {}
    for csv_path in csv_paths:
        algo_name, num_samples, temperature, _seed = parse_filename(csv_path)
        key = (algo_name, num_samples, temperature)
        grouped_paths.setdefault(key, []).append(csv_path)

    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 10
    output_dpi = 300

    fig, ax = plt.subplots(figsize=(10, 6))

    for (algo_name, num_samples, temperature), group_paths in grouped_paths.items():
        algo_marker = algo_marker_map.setdefault(
            algo_name, algo_markers[len(algo_marker_map) % len(algo_markers)]
        )
        algo_color = algorithm_color(algo_name)
        temp_style = temp_style_map.setdefault(
            temperature, temp_linestyles[len(temp_style_map) % len(temp_linestyles)]
        )
        group_rows = [load_csv(path) for path in group_paths]
        group_rows = [rows for rows in group_rows if rows]
        if not group_rows:
            continue
        fieldnames = group_rows[0][0].keys()

        for label, candidates in column_specs:
            column_name = resolve_column_name(fieldnames, candidates)
            if column_name is None:
                continue
            series_list = [
                _series_by_iteration(rows, column_name) for rows in group_rows
            ]
            iterations, means, stds = _aggregate_series(series_list)
            if not iterations:
                continue
            series_label = f"{display_algo_name(algo_name)}"
            if args.population_size:
                series_label += f" | population size={num_samples}"
            if args.include_temperature:
                series_label += f" | T={temperature}"
            series_label = f"{series_label} | {label}"
            if len(series_list) > 1:
                ax.fill_between(
                    iterations,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=algo_color,
                    alpha=0.2,
                    linewidth=0,
                )
            ax.plot(
                iterations,
                means,
                marker=algo_marker,
                linestyle=temp_style,
                color=algo_color,
                markersize=5,
                linewidth=1.6,
                label=series_label,
            )

    ax.set_xlabel("Iteration", fontsize=label_fontsize)
    ax.set_ylabel("Cost".replace("_", " "), fontsize=label_fontsize)
    if not args.no_log_scale:
        ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=legend_fontsize, ncol=1, frameon=False)
    fig.tight_layout()

    output_path = args.output or (folder / "compare.pdf")
    fig.savefig(output_path, dpi=output_dpi)
    plt.close(fig)

    for label, candidates in column_specs:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False
        for (algo_name, num_samples, temperature), group_paths in grouped_paths.items():
            algo_marker = algo_marker_map.setdefault(
                algo_name, algo_markers[len(algo_marker_map) % len(algo_markers)]
            )
            algo_color = algorithm_color(algo_name)
            temp_style = temp_style_map.setdefault(
                temperature, temp_linestyles[len(temp_style_map) % len(temp_linestyles)]
            )
            group_rows = [load_csv(path) for path in group_paths]
            group_rows = [rows for rows in group_rows if rows]
            if not group_rows:
                continue
            fieldnames = group_rows[0][0].keys()
            column_name = resolve_column_name(fieldnames, candidates)
            if column_name is None:
                continue
            series_list = [
                _series_by_iteration(rows, column_name) for rows in group_rows
            ]
            iterations, means, stds = _aggregate_series(series_list)
            if not iterations:
                continue
            series_label = f"{display_algo_name(algo_name)}"
            if args.population_size:
                series_label += f" | population size={num_samples}"
            if args.include_temperature:
                series_label += f" | T={temperature}"
            if len(series_list) > 1:
                ax.fill_between(
                    iterations,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=algo_color,
                    alpha=0.2,
                    linewidth=0,
                )
            ax.plot(
                iterations,
                means,
                marker=algo_marker,
                linestyle=temp_style,
                color=algo_color,
                markersize=5,
                linewidth=1.6,
                label=series_label,
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Iteration", fontsize=label_fontsize)
        ax.set_ylabel(label.replace("_", " "), fontsize=label_fontsize)
        if not args.no_log_scale:
            ax.set_yscale("log")
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.legend(fontsize=legend_fontsize, ncol=1, frameon=False)
        fig.tight_layout()
        column_output = folder / f"compare_{sanitize_filename(label)}.pdf"
        fig.savefig(column_output, dpi=output_dpi)
        plt.close(fig)


if __name__ == "__main__":
    main()
