from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import json


def _pick_latest_dir_with_params(
    experiments_dir: Path,
    algorithm_name: str | None,
    task_slug: str | None,
) -> Path:
    candidates = sorted([p for p in experiments_dir.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No experiments found under {experiments_dir}")
    for experiment_path in reversed(candidates):
        params_dir = _resolve_params_dir_from_experiment(
            experiment_path, algorithm_name, task_slug
        )
        if params_dir is not None:
            return params_dir
    raise FileNotFoundError(f"No params dirs found under {experiments_dir}")


def _resolve_params_dir_from_experiment(
    experiment_path: Path, algorithm_name: str | None, task_slug: str | None
) -> Path | None:
    if experiment_path.name.startswith("compare_"):
        if algorithm_name is not None:
            candidate = experiment_path / algorithm_name
            if (candidate / "params").exists() and _run_meta_matches(
                candidate, task_slug, algorithm_name
            ):
                return candidate / "params"
            return None
        subdirs = sorted([p for p in experiment_path.iterdir() if p.is_dir()])
        with_params = [
            p
            for p in subdirs
            if (p / "params").exists() and _run_meta_matches(p, task_slug, None)
        ]
        if with_params:
            return with_params[-1] / "params"
        return None
    if (experiment_path / "params").exists() and _run_meta_matches(
        experiment_path, task_slug, algorithm_name
    ):
        return experiment_path / "params"
    return None


def resolve_params_dir(
    base_dir: str | Path | None, algorithm_name: str | None, task_slug: str | None
) -> Path:
    if base_dir is None:
        base_dir = Path("tmp/experiments")
        if not base_dir.exists():
            raise FileNotFoundError(
                "No experiments found: tmp/experiments does not exist"
            )
    else:
        base_dir = Path(base_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"Params base dir not found: {base_dir}")

    if (base_dir / "params_latest.npy").exists() or list(
        base_dir.glob("params_iter_*.npy")
    ):
        return base_dir
    if (base_dir / "params").exists():
        return base_dir / "params"
    if base_dir.name.startswith("compare_") or base_dir.parent.name == "experiments":
        params_dir = _resolve_params_dir_from_experiment(
            base_dir, algorithm_name, task_slug
        )
        if params_dir is None:
            raise FileNotFoundError(f"No params dirs found under {base_dir}")
        return params_dir
    if base_dir == Path("tmp/experiments"):
        return _pick_latest_dir_with_params(base_dir, algorithm_name, task_slug)

    raise FileNotFoundError(f"Params dir not found under {base_dir}")


def resolve_params_file(
    params_dir: Path, iteration: int | None
) -> tuple[Path, int | None]:
    if iteration is not None:
        return params_dir / f"params_iter_{iteration:05d}.npy", iteration

    latest_path = params_dir / "params_latest.npy"
    if latest_path.exists():
        latest_iter_path = params_dir / "params_latest_iter.txt"
        latest_iter = None
        if latest_iter_path.exists():
            try:
                latest_iter = int(latest_iter_path.read_text(encoding="utf-8").strip())
            except ValueError:
                latest_iter = None
        return latest_path, latest_iter

    candidates = sorted(params_dir.glob("params_iter_*.npy"))
    if candidates:
        return candidates[-1], None
    raise FileNotFoundError(f"No params files found under {params_dir}")


def _unwrap_loaded_params(loaded: Any) -> Any:
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        return loaded.item()
    return loaded


def _convert_key_data(value: Any) -> Any:
    if not isinstance(value, np.ndarray):
        return value
    if value.dtype != np.uint32 or value.ndim != 1 or value.size not in (2, 4):
        return value
    wrap_key_data = getattr(jax.random, "wrap_key_data", None)
    if wrap_key_data is None:
        return value
    return wrap_key_data(value)


def _run_meta_matches(
    run_dir: Path,
    task_slug: str | None,
    algorithm_name: str | None,
) -> bool:
    if task_slug is None and algorithm_name is None:
        return True
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    if task_slug is not None and meta.get("task") != task_slug:
        return False
    if algorithm_name is not None and meta.get("controller") != algorithm_name:
        return False
    return True


def load_persisted_params(
    base_dir: str | Path | None,
    algorithm_name: str | None,
    task_slug: str | None,
    iteration: int | None,
) -> tuple[Any | None, int | None]:
    params_dir = resolve_params_dir(base_dir, algorithm_name, task_slug)
    file_path, resolved_iter = resolve_params_file(params_dir, iteration)
    print(f"Loading persisted params from {file_path}")
    if not file_path.exists():
        return None, None
    loaded = np.load(file_path, allow_pickle=True)
    loaded = _unwrap_loaded_params(loaded)
    loaded = jtu.tree_map(jnp.asarray, loaded)
    loaded = jtu.tree_map(_convert_key_data, loaded)
    return loaded, resolved_iter


def merge_loaded_params(initial_params: Any, loaded_params: Any) -> Any:
    if loaded_params is None:
        return initial_params
    if jtu.tree_structure(initial_params) == jtu.tree_structure(loaded_params):
        loaded_params = jtu.tree_map(
            _swap_prng_leaf, initial_params, loaded_params
        )
    if hasattr(loaded_params, "replace") and hasattr(initial_params, "rng"):
        try:
            return loaded_params.replace(rng=initial_params.rng)
        except Exception:
            return loaded_params
    return loaded_params


def _swap_prng_leaf(init_leaf: Any, loaded_leaf: Any) -> Any:
    init_dtype = getattr(init_leaf, "dtype", None)
    if init_dtype is not None and "key" in str(init_dtype):
        return init_leaf
    return loaded_leaf


def patch_init_params_from_disk(
    controller: Any,
    *,
    base_dir: str | Path | None,
    algorithm_name: str | None,
    task_slug: str | None,
    iteration: int | None,
) -> None:
    original_init = controller.init_params

    def _describe_tree(tree: Any) -> str:
        try:
            leaves = jtu.tree_leaves(tree)
        except Exception:
            return "<unprintable>"
        shapes = []
        for leaf in leaves:
            shape = getattr(leaf, "shape", None)
            if shape is None:
                shapes.append("<no-shape>")
            else:
                shapes.append(str(tuple(shape)))
        return ", ".join(shapes)

    def _describe_tree_details(tree: Any) -> str:
        try:
            leaves = jtu.tree_leaves(tree)
        except Exception:
            return "<unprintable>"
        details = []
        for leaf in leaves:
            shape = getattr(leaf, "shape", None)
            dtype = getattr(leaf, "dtype", None)
            details.append(f"{type(leaf).__name__}[{shape},{dtype}]")
        return ", ".join(details)

    def _wrapped_init_params(*args, **kwargs):
        params = original_init(*args, **kwargs)
        loaded_params, loaded_iter = load_persisted_params(
            base_dir, algorithm_name, task_slug, iteration
        )
        if loaded_params is None:
            return params
        print(
            "Loaded params shapes: "
            f"{_describe_tree(loaded_params)} | init shapes: {_describe_tree(params)}"
        )
        print(
            "Loaded params leaf details: "
            f"{_describe_tree_details(loaded_params)} | init leaf details: {_describe_tree_details(params)}"
        )
        setattr(controller, "loaded_params_iteration", loaded_iter)
        merged = merge_loaded_params(params, loaded_params)
        print(
            "Merged params leaf details: "
            f"{_describe_tree_details(merged)}"
        )
        return merged

    controller.init_params = _wrapped_init_params
