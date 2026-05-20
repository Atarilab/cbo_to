import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


def parse_filename(path: Path) -> Tuple[str, str, str]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Expected filename like <algo>_<num_samples>_<temperature>.csv, got {path.name}"
        )
    temperature = parts[-1]
    num_samples = parts[-2]
    algo_name = "_".join(parts[:-2])
    return algo_name, num_samples, temperature


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot costs from CSV files named <algo>_<num_samples>_<temperature>.csv"
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
        "--logscale",
        action="store_true",
        help="Plot costs on a log scale.",
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
    sample_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    temp_linestyles = ["-", "--", ":", "-."]
    algo_marker_map: Dict[str, str] = {}
    sample_color_map: Dict[str, str] = {}
    temp_style_map: Dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(10, 6))

    for csv_path in csv_paths:
        algo_name, num_samples, temperature = parse_filename(csv_path)
        algo_marker = algo_marker_map.setdefault(
            algo_name, algo_markers[len(algo_marker_map) % len(algo_markers)]
        )
        sample_color = sample_color_map.setdefault(
            num_samples, sample_colors[len(sample_color_map) % len(sample_colors)]
        )
        temp_style = temp_style_map.setdefault(
            temperature, temp_linestyles[len(temp_style_map) % len(temp_linestyles)]
        )

        rows = load_csv(csv_path)
        if not rows:
            continue
        fieldnames = rows[0].keys()

        iterations = []
        for row in rows:
            try:
                iterations.append(int(float(row.get("iteration", "0"))))
            except ValueError:
                iterations.append(0)

        for label, candidates in column_specs:
            column_name = resolve_column_name(fieldnames, candidates)
            if column_name is None:
                continue
            values = []
            for row in rows:
                try:
                    values.append(float(row[column_name]))
                except (KeyError, ValueError, TypeError):
                    values.append(float("nan"))

            series_label = (
                f"{algo_name} | n={num_samples} | T={temperature} | {label}"
            )
            ax.plot(
                iterations,
                values,
                marker=algo_marker,
                linestyle=temp_style,
                color=sample_color,
                markersize=4,
                linewidth=1.2,
                label=series_label,
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    if args.logscale:
        ax.set_yscale("log")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=8, ncol=1, frameon=False)
    fig.tight_layout()

    output_path = args.output or (folder / "compare.pdf")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    for label, candidates in column_specs:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False
        for csv_path in csv_paths:
            algo_name, num_samples, temperature = parse_filename(csv_path)
            algo_marker = algo_marker_map.setdefault(
                algo_name, algo_markers[len(algo_marker_map) % len(algo_markers)]
            )
            sample_color = sample_color_map.setdefault(
                num_samples, sample_colors[len(sample_color_map) % len(sample_colors)]
            )
            temp_style = temp_style_map.setdefault(
                temperature, temp_linestyles[len(temp_style_map) % len(temp_linestyles)]
            )

            rows = load_csv(csv_path)
            if not rows:
                continue
            fieldnames = rows[0].keys()
            column_name = resolve_column_name(fieldnames, candidates)
            if column_name is None:
                continue

            iterations = []
            values = []
            for row in rows:
                try:
                    iterations.append(int(float(row.get("iteration", "0"))))
                except ValueError:
                    iterations.append(0)
                try:
                    values.append(float(row[column_name]))
                except (KeyError, ValueError, TypeError):
                    values.append(float("nan"))

            series_label = f"{algo_name} | n={num_samples} | T={temperature}"
            ax.plot(
                iterations,
                values,
                marker=algo_marker,
                linestyle=temp_style,
                color=sample_color,
                markersize=4,
                linewidth=1.2,
                label=series_label,
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Iteration")
        ax.set_ylabel(label)
        if args.logscale:
            ax.set_yscale("log")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.legend(fontsize=8, ncol=1, frameon=False)
        fig.tight_layout()
        column_output = folder / f"compare_{sanitize_filename(label)}.pdf"
        fig.savefig(column_output, dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
