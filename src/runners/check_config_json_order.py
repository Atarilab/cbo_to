#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import List


def _load_keys(path: Path) -> List[str]:
    with path.open("r") as handle:
        data = json.load(handle)
    return list(data.keys())


def main() -> int:
    config_dir = Path(__file__).resolve().parents[2] / "config"
    paths = sorted(config_dir.rglob("*.json"))
    if not paths:
        print("No JSON files found.")
        return 1

    base_path = paths[0]
    base_keys = _load_keys(base_path)
    mismatches = []

    for path in paths[1:]:
        keys = _load_keys(path)
        if keys != base_keys:
            mismatches.append(path)

    if mismatches:
        print(f"Mismatch vs {base_path}:")
        for path in mismatches:
            print(f"- {path}")
        return 1

    print(f"All {len(paths)} JSON files match key order and entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
