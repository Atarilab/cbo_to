from __future__ import annotations

import argparse
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np

from algs.cbo_x import CBOx

try:
    from pcbo.cbox_polarized import CBOxPolarized
except ImportError:
    CBOxPolarized = None
from utils.config_loader import load_compare_config
from utils.mjx_utils import apply_mjx_settings
from utils.joint_plot_utils import (
    get_leg_actuator_indices,
    get_leg_qpos_indices,
    get_leg_qvel_indices,
)
from runners.particle_replay import (
    compute_costs,
    load_costs,
    resolve_experiment_dir,
    resolve_particles_path,
    select_knots,
    select_knots_by_rank,
)
from trajopt.optimizer import TrajectoryOptimizer


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


def _read_task_from_metadata(
    experiment_dir: str | Path | None,
    *,
    algorithm: str | None,
) -> str | None:
    source = resolve_experiment_dir(experiment_dir, algorithm=algorithm)
    meta_path = source.experiment_dir / "run_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    task = meta.get("task")
    if task in {"humanoid", "double_cartpole", "cartpole"}:
        return task
    if task == "d-cartpole":
        return "double_cartpole"
    return None


def _resolve_task_for_plot(
    task_name: str | None,
    *,
    experiment_dir: str | Path | None,
    algorithm: str | None,
) -> str:
    if task_name is not None:
        return task_name
    inferred = _read_task_from_metadata(experiment_dir, algorithm=algorithm)
    if inferred is not None:
        return inferred
    raise ValueError(
        "Task not provided and no task metadata found; pass --task explicitly."
    )


def _resolve_config_for_plot(
    config_path: str | Path | None,
    *,
    experiment_dir: str | Path | None,
    algorithm: str | None,
) -> Path:
    if config_path is not None:
        return Path(config_path)

    source = resolve_experiment_dir(experiment_dir, algorithm=algorithm)
    meta_path = source.experiment_dir / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        config_name = meta.get("config")
        if isinstance(config_name, str) and config_name.strip():
            candidate = source.experiment_dir / config_name
            if candidate.exists():
                return candidate

    json_candidates = sorted(
        p for p in source.experiment_dir.glob("*.json") if p.name != "run_meta.json"
    )
    if len(json_candidates) == 1:
        return json_candidates[0]

    if json_candidates:
        names = ", ".join(p.name for p in json_candidates[:5])
        raise FileNotFoundError(
            "Could not determine config automatically from run metadata. "
            f"Multiple JSON candidates found in {source.experiment_dir}: {names}. "
            "Pass --config explicitly."
        )

    raise FileNotFoundError(
        "Could not determine config automatically. "
        f"No config reference in {meta_path} and no config JSON found in {source.experiment_dir}. "
        "Pass --config explicitly."
    )


def _parse_dims(value: str | None, size: int, max_dims: int) -> list[int]:
    if value is None:
        return list(range(min(size, max_dims)))
    value = value.strip().lower()
    if value in {"all", "*"}:
        return list(range(size))
    dims = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        dims.append(int(part))
    return [d for d in dims if 0 <= d < size]


def _setting(settings: dict[str, object], *keys: str, default):
    for key in keys:
        if key in settings:
            return settings[key]
    return default


def _build_optimizer(
    task_name: str,
    config_path: Path,
    *,
    algorithm_name: str | None = None,
) -> TrajectoryOptimizer:
    config = load_compare_config(config_path)
    settings = config.settings_for_algorithm(algorithm_name)
    import mujoco
    from tasks.cart_pole import CartPole
    from tasks.double_cartpole import DoubleCartPoleUnconstrained
    from tasks.humanoid import Humanoid

    if task_name == "humanoid":
        task = Humanoid(
            reference_filename="DefaultDatasets/mocap/UnitreeG1/balance.npz",
            model_xml_path="models/g1/scene_23dof.xml",
            start_frame=0,
            flag_zero_running_cost=True,
        )
        apply_mjx_settings(task, config.mjx_integration_time)
        mj_model = task.mj_model
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = task.start_config
    elif task_name == "double_cartpole":
        task = DoubleCartPoleUnconstrained()
        mj_model = task.mj_model
        mj_data = mujoco.MjData(mj_model)
    elif task_name == "cartpole":
        task = CartPole()
        mj_model = task.mj_model
        mj_data = mujoco.MjData(mj_model)
    else:
        raise ValueError("task must be one of: humanoid, double_cartpole, cartpole")

    controller_name = (
        algorithm_name if algorithm_name in {"CBOx", "CBOxPolarized"} else "CBOx"
    )
    if controller_name == "CBOxPolarized" and CBOxPolarized is None:
        raise ImportError("CBOxPolarized requires the 'pcbo' package which is not installed.")
    controller_cls = CBOxPolarized if controller_name == "CBOxPolarized" else CBOx
    ctrl = controller_cls(
        task=task,
        num_samples=int(_setting(settings, "num_samples", default=config.num_samples)),
        noise_level=_setting(settings, "noise_level", default=config.noise_level),
        temperature=_setting(settings, "temperature", default=config.temperature),
        plan_horizon=config.horizon,
        num_knots=config.num_knots,
        spline_type=config.spline_type,
        iterations=1,
        lambda_=_setting(settings, "decay", "cbo_decay", default=config.cbo_decay),
        time_step=_setting(settings, "dt", "cbo_dt", default=config.cbo_dt),
        flag_anistropic=_setting(
            settings, "anisotropic", "flag_anistropic", default=config.flag_anistropic
        ),
        flag_multiply_cost=_setting(
            settings, "multiply_cost", "flag_multiply_cost", default=config.flag_multiply_cost
        ),
        flag_multiply_knot_weights=_setting(
            settings,
            "multiply_knot_weights",
            "flag_multiply_knot_weights",
            default=config.flag_multiply_knot_weights,
        ),
        limit_temp=_setting(
            settings, "limit_temp", "cbo_limit_temp", default=config.cbo_limit_temp
        ),
    )
    ctrl.temp_decay = _setting(
        settings, "temp_decay", "cbo_temp_decay", default=config.cbo_temp_decay
    )
    return TrajectoryOptimizer(
        controller_name,
        ctrl,
        mj_model,
        mj_data,
        max_eval=config.max_eval,
        config_path=config_path,
    )


def _load_knots(
    optimizer: TrajectoryOptimizer,
    *,
    experiment_dir: str | Path | None,
    algorithm: str | None,
    iteration: int | None,
    rank: int | None,
    particle_index: int | None,
    use_consensus: bool,
) -> tuple[jnp.ndarray, str, str]:
    which = "iter" if iteration is not None else "latest"
    source = resolve_experiment_dir(experiment_dir, algorithm=algorithm)

    if use_consensus:
        if which == "latest":
            consensus_path = source.particles_dir / "consensus_latest.npy"
            if not consensus_path.exists():
                candidates = sorted(source.particles_dir.glob("consensus_iter_*.npy"))
                if not candidates:
                    raise FileNotFoundError(
                        f"Consensus files not found under {source.particles_dir}"
                    )
                consensus_path = candidates[-1]
            iter_tag = _infer_iteration_tag(consensus_path)
        else:
            if iteration is None:
                raise ValueError("iteration must be provided when which='iter'")
            consensus_path = (
                source.particles_dir / f"consensus_iter_{iteration:05d}.npy"
            )
            if not consensus_path.exists():
                raise FileNotFoundError(f"Consensus file not found: {consensus_path}")
            iter_tag = f"iter_{iteration:05d}"
        knots = jnp.asarray(np.load(consensus_path))
        label = "consensus"
        return knots, iter_tag, label

    particles_path = resolve_particles_path(source, which=which, iteration=iteration)
    particles = np.load(particles_path)
    iter_tag = _infer_iteration_tag(particles_path)

    if particle_index is not None:
        knots = select_knots(particles, particle_index=particle_index)
        label = f"idx_{particle_index:04d}"
        return knots, iter_tag, label

    if rank is None:
        rank = 0

    try:
        costs = load_costs(source, which=which, iteration=iteration)
    except FileNotFoundError:
        costs = compute_costs(optimizer, particles)
    knots, idx, cost = select_knots_by_rank(particles, costs, rank=rank)
    label = f"rank_{rank:04d}_idx_{idx:04d}_cost_{cost:.6g}"
    return knots, iter_tag, label


def _load_ranked_knots(
    optimizer: TrajectoryOptimizer,
    *,
    experiment_dir: str | Path | None,
    algorithm: str | None,
    iteration: int | None,
    max_rank: int,
) -> tuple[list[tuple[jnp.ndarray, str]], str]:
    which = "iter" if iteration is not None else "latest"
    source = resolve_experiment_dir(experiment_dir, algorithm=algorithm)
    particles_path = resolve_particles_path(source, which=which, iteration=iteration)
    particles = np.load(particles_path)
    iter_tag = _infer_iteration_tag(particles_path)

    try:
        costs = load_costs(source, which=which, iteration=iteration)
    except FileNotFoundError:
        costs = compute_costs(optimizer, particles)
    order = np.argsort(np.asarray(costs).reshape(-1))
    max_rank = max(0, min(int(max_rank), order.shape[0] - 1))
    selected = []
    for rank, idx in enumerate(order[: max_rank + 1]):
        cost = float(costs[int(idx)])
        label = f"rank_{rank:04d}_idx_{idx:04d}_cost_{cost:.6g}"
        selected.append((jnp.asarray(particles[int(idx)]), label))
    return selected, iter_tag


def _plot_states_controls(
    qpos: np.ndarray,
    qvel: np.ndarray,
    controls: np.ndarray,
    dt: float,
    *,
    qpos_dims: list[int],
    qvel_dims: list[int],
    ctrl_dims: list[int],
    out_path: Path,
    show: bool,
) -> None:
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = []
    for dim in qpos_dims:
        series.append(("qpos", dim, qpos[:, dim]))
    for dim in qvel_dims:
        series.append(("qvel", dim, qvel[:, dim]))
    for dim in ctrl_dims:
        series.append(("u", dim, controls[:, dim]))

    rows = max(1, len(series))
    fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(9, 2.2 * rows))
    if rows == 1:
        axes = [axes]

    t = np.arange(controls.shape[0]) * dt
    for ax, (name, dim, values) in zip(axes, series):
        ax.plot(t, values, linewidth=1.2)
        ax.set_ylabel(f"{name}[{dim}]")
        ax.grid(True)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot rollout states (qpos/qvel) alongside controls."
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
        "--rank",
        type=int,
        default=None,
        help="Plot ranks 0..K (inclusive). Default: 0.",
    )
    parser.add_argument(
        "--particle-index",
        type=int,
        default=None,
        help="Select a particle by index (overrides rank).",
    )
    parser.add_argument(
        "--consensus",
        action="store_true",
        help="Plot the consensus knots instead of a particle.",
    )
    parser.add_argument(
        "--task",
        choices=["humanoid", "double_cartpole", "cartpole"],
        default=None,
        help="Task to simulate (defaults to value saved in the experiment dir).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config JSON used to build the controller (default: auto-detect from the experiment dir).",
    )
    parser.add_argument(
        "--qpos",
        type=str,
        default=None,
        help="Comma-separated qpos indices (e.g. '0,1,2') or 'all'.",
    )
    parser.add_argument(
        "--qvel",
        type=str,
        default=None,
        help="Comma-separated qvel indices (e.g. '0,1,2') or 'all'.",
    )
    parser.add_argument(
        "--ctrl",
        type=str,
        default=None,
        help="Comma-separated control indices (e.g. '0,1') or 'all'.",
    )
    parser.add_argument(
        "--max-dims",
        type=int,
        default=6,
        help="Max dimensions to plot per signal when indices are not specified.",
    )
    parser.add_argument(
        "--legs-only",
        action="store_true",
        help="Plot only leg joints (humanoid task only).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <experiment_dir>/plot_states_controls).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (also saves PNGs).",
    )
    args = parser.parse_args()

    task_name = _resolve_task_for_plot(
        args.task, experiment_dir=args.experiment_dir, algorithm=args.algorithm
    )
    config_path = _resolve_config_for_plot(
        args.config, experiment_dir=args.experiment_dir, algorithm=args.algorithm
    )
    optimizer = _build_optimizer(task_name, config_path, algorithm_name=args.algorithm)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is None:
        source = resolve_experiment_dir(args.experiment_dir, algorithm=args.algorithm)
        out_dir = source.experiment_dir / "plot_states_controls"

    if args.consensus or args.particle_index is not None:
        knots, iter_tag, label = _load_knots(
            optimizer,
            experiment_dir=args.experiment_dir,
            algorithm=args.algorithm,
            iteration=args.iteration,
            rank=args.rank,
            particle_index=args.particle_index,
            use_consensus=args.consensus,
        )
        knots_list = [(knots, label)]
    else:
        max_rank = args.rank if args.rank is not None else 0
        knots_list, iter_tag = _load_ranked_knots(
            optimizer,
            experiment_dir=args.experiment_dir,
            algorithm=args.algorithm,
            iteration=args.iteration,
            max_rank=max_rank,
        )

    for knots, label in knots_list:
        optimizer.reset_mjx_data()
        state = optimizer.mjx_data
        knots = jnp.asarray(knots)
        if knots.ndim == 2:
            knots_batch = knots[None, ...]
        else:
            raise ValueError(f"Expected knots with 2 dims, got {knots.shape}")
        controls = optimizer.knots2ctrls(knots_batch)
        states, _ = optimizer.controller.eval_rollouts(
            optimizer.controller.model, state, controls, knots_batch
        )

        states_np = jax.device_get(states)
        qpos = np.asarray(states_np.qpos)[0]
        qvel = np.asarray(states_np.qvel)[0]
        controls_np = np.asarray(jax.device_get(controls))[0]

        qpos_dims = _parse_dims(args.qpos, qpos.shape[1], args.max_dims)
        qvel_dims = _parse_dims(args.qvel, qvel.shape[1], args.max_dims)
        ctrl_dims = _parse_dims(args.ctrl, controls_np.shape[1], args.max_dims)
        if args.legs_only and task_name == "humanoid":
            if args.qpos is None:
                qpos_dims = [
                    d
                    for d in get_leg_qpos_indices(optimizer.mj_model)
                    if d < qpos.shape[1]
                ]
            if args.qvel is None:
                qvel_dims = [
                    d
                    for d in get_leg_qvel_indices(optimizer.mj_model)
                    if d < qvel.shape[1]
                ]
            if args.ctrl is None:
                ctrl_dims = [
                    d
                    for d in get_leg_actuator_indices(optimizer.mj_model)
                    if d < controls_np.shape[1]
                ]

        out_path = out_dir / f"states_controls_{iter_tag}_{label}.pdf"

        _plot_states_controls(
            qpos,
            qvel,
            controls_np,
            float(optimizer.controller.dt),
            qpos_dims=qpos_dims,
            qvel_dims=qvel_dims,
            ctrl_dims=ctrl_dims,
            out_path=out_path,
            show=args.show,
        )


if __name__ == "__main__":
    main()
