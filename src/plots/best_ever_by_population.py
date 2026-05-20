import argparse
import csv
import math
import statistics
from bisect import bisect_right
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


def resolve_column_name(fieldnames: Iterable[str], candidates: List[str]) -> str | None:
    field_set = {name for name in fieldnames}
    for candidate in candidates:
        if candidate in field_set:
            return candidate
    return None


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


def _value_at_or_before(
    series: Dict[int, float], iterations: List[int], target_iter: float
) -> float | None:
    idx = bisect_right(iterations, target_iter) - 1
    if idx < 0:
        return None
    value = series[iterations[idx]]
    if not math.isfinite(value):
        return None
    return value


def _parse_population_size(folder: Path) -> int | None:
    if not folder.name.startswith("particles_"):
        return None
    suffix = folder.name.split("particles_", 1)[1]
    return int(suffix) if suffix.isdigit() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare best-ever metrics across population sizes. Expects folders like "
            "<root>/particles_<num>/*.csv with filenames <algo>_<num_samples>_<temperature>.csv."
        )
    )
    parser.add_argument("folder", type=Path, help="Folder containing particles_* folders")
    parser.add_argument(
        "--column",
        type=str,
        help="Column to plot (default: best_ever_loss if present, else best_ever_cost).",
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
        "--num-xticks",
        type=int,
        default=6,
        help="Target number of x-axis ticks (default: 6).",
    )
    parser.add_argument(
        "--xtick-step",
        type=int,
        help="Explicit x-axis tick step in samples (overrides --num-xticks).",
    )
    parser.add_argument(
        "--xtick-k",
        action="store_true",
        default=True,
        help="Format x-axis ticks in thousands (e.g., 4k).",
    )
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    particle_dirs: List[Tuple[int, Path]] = []
    for child in sorted(p for p in folder.iterdir() if p.is_dir()):
        pop_size = _parse_population_size(child)
        if pop_size is not None:
            particle_dirs.append((pop_size, child))

    if not particle_dirs:
        raise SystemExit(f"No particles_* folders found under {folder}")

    particle_dirs.sort(key=lambda item: item[0])
    max_population = max(pop_size for pop_size, _ in particle_dirs)

    grouped_series: Dict[Tuple[str, int], List[Dict[int, float]]] = {}
    column_name_per_group: Dict[Tuple[str, int], str] = {}

    for pop_size, pop_dir in particle_dirs:
        csv_paths = sorted(pop_dir.glob("*.csv"))
        for csv_path in csv_paths:
            algo_name, _num_samples, _temperature, _seed = parse_filename(csv_path)
            rows = load_csv(csv_path)
            if not rows:
                continue
            fieldnames = rows[0].keys()
            candidates = (
                [args.column]
                if args.column
                else ["best_ever_loss", "best_ever_cost"]
            )
            column_name = resolve_column_name(fieldnames, candidates)
            if column_name is None:
                continue
            column_name_per_group[(algo_name, pop_size)] = column_name
            series = _series_by_iteration(rows, column_name)
            grouped_series.setdefault((algo_name, pop_size), []).append(series)

    if not grouped_series:
        raise SystemExit(f"No usable CSV files found under {folder}")

    pop_linestyles = ["-", "--", ":", "-."]
    pop_style_map: Dict[int, str] = {}

    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 10
    output_dpi = 300

    fig, ax = plt.subplots(figsize=(10, 6))

    global_max_samples = 0
    for (algo_name, pop_size), series_list in grouped_series.items():
        max_iter = max((max(series.keys()) for series in series_list if series), default=0)
        global_max_samples = max(global_max_samples, max_iter * pop_size)

    for (algo_name, pop_size), series_list in grouped_series.items():
        if not series_list:
            continue
        algo_color = algorithm_color(algo_name)
        pop_style = pop_style_map.setdefault(
            pop_size, pop_linestyles[len(pop_style_map) % len(pop_linestyles)]
        )
        series_iters = [sorted(series.keys()) for series in series_list]
        max_iter = max((max(iters) for iters in series_iters if iters), default=0)
        max_samples = max_iter * pop_size
        if max_samples < max_population:
            continue

        sample_counts: List[int] = []
        means: List[float] = []
        stds: List[float] = []
        for sample_count in range(max_population, max_samples + 1, max_population):
            target_iter = sample_count / pop_size
            values: List[float] = []
            for series, iterations in zip(series_list, series_iters):
                value = _value_at_or_before(series, iterations, target_iter)
                if value is not None:
                    values.append(value)
            if not values:
                continue
            sample_counts.append(sample_count)
            means.append(statistics.fmean(values))
            stds.append(statistics.pstdev(values) if len(values) > 1 else 0.0)

        if not sample_counts:
            continue

        label = f"{display_algo_name(algo_name)} | particles={pop_size}"
        if any(std > 0 for std in stds):
            ax.fill_between(
                sample_counts,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                color=algo_color,
                alpha=0.2,
                linewidth=0,
            )
        ax.plot(
            sample_counts,
            means,
            linestyle=pop_style,
            color=algo_color,
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel("Number of samples", fontsize=label_fontsize)
    y_label = column_name_per_group.get(next(iter(column_name_per_group)), "best_ever_cost")
    ax.set_ylabel(y_label.replace("_", " "), fontsize=label_fontsize)
    if not args.no_log_scale:
        ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    if global_max_samples >= max_population:
        if args.xtick_step is not None and args.xtick_step > 0:
            step = args.xtick_step
            start = step
        else:
            # Use a coarse tick step to avoid dense labels for large sample counts.
            raw_step = max_population
            max_ticks = max(args.num_xticks, 2)
            step = max(raw_step, int(math.ceil(global_max_samples / (max_ticks - 1))))
            # Round step up to a multiple of max_population for aligned ticks.
            if step % max_population != 0:
                step = int(math.ceil(step / max_population) * max_population)
            start = max_population
        ax.set_xticks(list(range(start, global_max_samples + 1, step)))
        def _format_tick(x: float) -> str:
            if args.xtick_k:
                base = 1000 if abs(x) < 1_000_000 else 1_000_000
                rounded = int(round(x / base) * base)
                return f"{rounded // 1000}k"
            rounded = int(round(x / 1000.0) * 1000)
            return f"{rounded:,}"

        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _pos: _format_tick(x)))
    ax.legend(fontsize=legend_fontsize, ncol=1, frameon=False)
    fig.tight_layout()

    output_path = args.output or (folder / "compare_best_ever_by_population.pdf")
    fig.savefig(output_path, dpi=output_dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
