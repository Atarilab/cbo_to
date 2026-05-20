import argparse
import json
import re
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Pattern, Tuple


_DEFAULT_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_-]*_[0-9]+_(?:unknown|[0-9]+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?e[+-]?[0-9]+)\.csv$"
)


def _find_csvs(root: Path, pattern: Pattern[str]) -> list[Path]:
    candidates = [p for p in root.rglob("*.csv") if p.parent.name == "stats"]
    return [p for p in candidates if pattern.match(p.name)]


def _resolve_compare_date(path: Path) -> Optional[str]:
    # Expected layout: <root>/<compare_date>/<algorithm>/stats/<file>.csv
    try:
        return path.parents[2].name
    except IndexError:
        return None


def _parse_compare_date(value: str) -> Optional[date]:
    match = re.search(r"(\d{8})(?:_\d{6})?$", value)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _extract_particles(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _load_task_and_particles_from_json(
    path: Path,
    *,
    algorithm_name: Optional[str] = None,
) -> Tuple[Optional[str], Optional[int]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    task = None
    task_raw = data.get("task")
    if isinstance(task_raw, str) and task_raw.strip():
        task = task_raw.strip()
    elif isinstance(task_raw, dict):
        nested_task = task_raw.get("name", task_raw.get("task"))
        if isinstance(nested_task, str) and nested_task.strip():
            task = nested_task.strip()

    particles = _extract_particles(data.get("num_samples"))
    algorithm_defaults = data.get("algorithm_defaults")
    if particles is None and isinstance(algorithm_defaults, dict):
        particles = _extract_particles(algorithm_defaults.get("num_samples"))
    if algorithm_name is not None:
        algorithms = data.get("algorithms")
        if isinstance(algorithms, list):
            for raw_algorithm in algorithms:
                if isinstance(raw_algorithm, str):
                    continue
                if not isinstance(raw_algorithm, dict):
                    continue
                if raw_algorithm.get("name") != algorithm_name:
                    continue
                params = raw_algorithm.get("params")
                if isinstance(params, dict):
                    particles = _extract_particles(params.get("num_samples")) or particles
                particles = _extract_particles(raw_algorithm.get("num_samples")) or particles
                break
    return task, particles


def _resolve_task_and_particles(csv_path: Path) -> Tuple[Optional[str], Optional[int]]:
    # Expected layout: <root>/<compare_date>/<algorithm>/stats/<file>.csv
    # Prefer run_meta.json (and its config) in the algorithm folder, then fall back to other JSONs.
    try:
        algorithm_dir = csv_path.parents[1]
    except IndexError:
        return None, None
    task = None
    particles = None
    run_meta = algorithm_dir / "run_meta.json"
    algorithm_name = algorithm_dir.name
    if run_meta.exists():
        task_from_meta, _ = _load_task_and_particles_from_json(run_meta)
        if task_from_meta:
            task = task_from_meta
        try:
            run_meta_data = json.loads(run_meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            run_meta_data = {}
        controller = run_meta_data.get("controller")
        if isinstance(controller, str) and controller.strip():
            algorithm_name = controller.strip()
        config_name = run_meta_data.get("config")
        if isinstance(config_name, str) and config_name.strip():
            config_path = algorithm_dir / config_name
            if config_path.exists():
                task_from_config, particles_from_config = _load_task_and_particles_from_json(
                    config_path,
                    algorithm_name=algorithm_name,
                )
                if task is None and task_from_config:
                    task = task_from_config
                if particles is None and particles_from_config is not None:
                    particles = particles_from_config
        if task and particles is not None:
            return task, particles
    for json_path in sorted(algorithm_dir.glob("*.json")):
        task_from_json, particles_from_json = _load_task_and_particles_from_json(
            json_path,
            algorithm_name=algorithm_name,
        )
        if task is None and task_from_json:
            task = task_from_json
        if particles is None and particles_from_json is not None:
            particles = particles_from_json
        if task and particles is not None:
            break
    return task, particles


def _unique_path(base: Path) -> Path:
    if not base.exists():
        return base
    stem = base.stem
    suffix = base.suffix
    parent = base.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate compare CSVs into a common output folder with seed_<datetime> suffix."
        )
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=None,
        help="Output folder for aggregated CSV files.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("tmp/experiments"),
        help="Root folder to search for compare runs (default: tmp/experiments).",
    )
    parser.add_argument(
        "--compare-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "Specific compare_* directory to aggregate. Repeat to aggregate an exact "
            "set of runs instead of scanning --input-root."
        ),
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=_DEFAULT_PATTERN.pattern,
        help=(
            "Regex for input CSV filenames. Defaults to "
            "[algo]_[samples]_[temperature].csv with numeric/1e-16 formats."
        ),
    )
    parser.add_argument(
        "--no-pattern",
        action="store_true",
        help="Ignore --pattern and include only CSVs with at least one digit in the filename.",
    )
    parser.add_argument(
        "--last-days",
        type=int,
        default=None,
        help="Only include compare runs from the last N days (inclusive of today).",
    )
    args = parser.parse_args()

    input_root = args.input_root
    if args.compare_dir:
        for compare_dir in args.compare_dir:
            if not compare_dir.exists():
                raise SystemExit(f"Compare directory not found: {compare_dir}")
            if not compare_dir.is_dir():
                raise SystemExit(f"Compare path is not a directory: {compare_dir}")
    elif not input_root.exists():
        raise SystemExit(f"Input root not found: {input_root}")
    output_root = args.output
    if output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path("tmp") / f"agg_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.no_pattern:
        search_roots = args.compare_dir or [input_root]
        csv_paths = []
        for root in search_roots:
            csv_paths.extend(
                p
                for p in _find_csvs(root, re.compile(r".*"))
                if any(ch.isdigit() for ch in p.name)
            )
    else:
        try:
            filename_pattern = re.compile(args.pattern)
        except re.error as exc:
            raise SystemExit(f"Invalid --pattern regex: {exc}")
        search_roots = args.compare_dir or [input_root]
        csv_paths = []
        for root in search_roots:
            csv_paths.extend(_find_csvs(root, filename_pattern))
    if not csv_paths:
        raise SystemExit(f"No stats CSV files found under {input_root}")

    if args.last_days is not None and args.last_days < 1:
        raise SystemExit("--last-days must be >= 1")

    copied = 0
    skipped = 0
    skipped_date = 0
    missing_task = 0
    missing_particles = 0
    today = datetime.now().date() if args.last_days is not None else None
    start_date = (
        today - timedelta(days=args.last_days - 1) if today is not None else None
    )
    for csv_path in csv_paths:
        compare_date = _resolve_compare_date(csv_path)
        if not compare_date:
            skipped += 1
            continue
        if start_date is not None:
            compare_day = _parse_compare_date(compare_date)
            if compare_day is None or compare_day < start_date or compare_day > today:
                skipped_date += 1
                continue
        task_name, particles = _resolve_task_and_particles(csv_path)
        if not task_name:
            missing_task += 1
            continue
        if particles is None:
            missing_particles += 1
            continue
        stem = csv_path.stem
        target_name = f"{stem}_seed_{compare_date}.csv"
        target_dir = output_root / task_name / f"particles_{particles}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = _unique_path(target_dir / target_name)
        shutil.copy2(csv_path, target_path)
        copied += 1

    print(f"Copied {copied} files to {output_root}")
    if skipped:
        print(f"Skipped {skipped} files with unexpected paths")
    if skipped_date:
        print(f"Skipped {skipped_date} files outside --last-days window")
    if missing_task:
        print(f"Skipped {missing_task} files missing task metadata")
    if missing_particles:
        print(f"Skipped {missing_particles} files missing particle metadata")


if __name__ == "__main__":
    main()
