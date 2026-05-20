from __future__ import annotations

import json
from pathlib import Path


def infer_task_slug(task: object | None) -> str | None:
    if task is None:
        return None
    class_name = type(task).__name__.lower()
    if "double" in class_name and "cartpole" in class_name:
        return "double_cartpole"
    if "cartpole" in class_name:
        return "cartpole"
    if "humanoid" in class_name:
        return "humanoid"
    return None


def infer_algorithm_slug(controller: object | None, controller_name: str | None) -> str | None:
    if controller is not None:
        class_name = type(controller).__name__.lower()
        if "cboxpolarized" in class_name or ("cbox" in class_name and "polarized" in class_name):
            return "cbox_polarized"
        if "cbox" in class_name:
            return "cbox"
    if controller_name is None:
        return None
    name = controller_name.lower().replace("-", "_").replace(" ", "")
    if name == "cbox":
        return "cbox"
    if name in {"cbox_polarized", "cboxpolarized"}:
        return "cbox_polarized"
    return None


def write_run_metadata(
    run_dir: Path,
    *,
    controller_name: str,
    task_slug: str | None,
    config_name: str | None,
    algorithm_slug: str | None,
) -> None:
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        return
    meta: dict[str, str] = {"controller": controller_name}
    if task_slug is not None:
        meta["task"] = task_slug
    if config_name is not None:
        meta["config"] = config_name
    if algorithm_slug is not None:
        meta["algorithm"] = algorithm_slug
    meta_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
