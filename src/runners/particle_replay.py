from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from utils.config_loader import load_compare_config
from trajopt.optimizer import TrajectoryOptimizer
from trajopt.helpers import (
    compute_gif_playback_settings,
    build_stride_frame_indices,
)


@dataclass(frozen=True)
class ParticleSource:
    experiment_dir: Path
    particles_dir: Path


def resolve_experiment_dir(
    experiment_dir: str | Path | None = None,
    *,
    algorithm: str | None = None,
) -> ParticleSource:
    """
    Resolve an experiment directory created under `tmp/experiments/<timestamp>/`
    or `tmp/experiments/compare_<timestamp>/<algorithm>/`.

    If `experiment_dir` is None, picks the latest timestamp directory.
    If a compare run directory is selected, `algorithm` can choose the subfolder.
    """
    base_dir = Path("tmp/experiments")
    if experiment_dir is None:
        if not base_dir.exists():
            raise FileNotFoundError("No experiments found: tmp/experiments does not exist")
        candidates = sorted([p for p in base_dir.iterdir() if p.is_dir()])
        if not candidates:
            raise FileNotFoundError("No experiments found under tmp/experiments")
        experiment_path = candidates[-1]
    else:
        experiment_path = Path(experiment_dir)
        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment dir not found: {experiment_path}")

    if experiment_path.name.startswith("compare_"):
        if algorithm is not None:
            candidate = experiment_path / algorithm
            if not candidate.exists():
                raise FileNotFoundError(f"Algorithm dir not found: {candidate}")
            experiment_path = candidate
        else:
            subdirs = sorted([p for p in experiment_path.iterdir() if p.is_dir()])
            if not subdirs:
                raise FileNotFoundError(f"No algorithm subdirs found under {experiment_path}")
            with_particles = [p for p in subdirs if (p / "particles").exists()]
            if not with_particles:
                raise FileNotFoundError(f"No particles dirs found under {experiment_path}")
            experiment_path = with_particles[-1]

    particles_dir = experiment_path / "particles"
    if not particles_dir.exists():
        raise FileNotFoundError(f"Particles dir not found: {particles_dir}")
    return ParticleSource(experiment_dir=experiment_path, particles_dir=particles_dir)


def load_particles(
    source: ParticleSource,
    *,
    which: Literal["latest", "iter"] = "latest",
    iteration: int | None = None,
) -> np.ndarray:
    """
    Load particles saved by CBOx.

    Returns a numpy array shaped like `(num_particles, num_knots, nu)`.
    """
    file_path = resolve_particles_path(source, which=which, iteration=iteration)
    return np.load(file_path)


def resolve_particles_path(
    source: ParticleSource,
    *,
    which: Literal["latest", "iter"] = "latest",
    iteration: int | None = None,
) -> Path:
    """Resolve the particles file path without loading its contents."""
    if which == "latest":
        file_path = source.particles_dir / "particles_latest.npy"
        if not file_path.exists():
            candidates = sorted(source.particles_dir.glob("particles_iter_*.npy"))
            if not candidates:
                raise FileNotFoundError(f"Particles file not found: {file_path}")
            file_path = candidates[-1]
            print(f"particles_latest.npy missing; using {file_path.name}")
    else:
        if iteration is None:
            raise ValueError("iteration must be provided when which='iter'")
        file_path = source.particles_dir / f"particles_iter_{iteration:05d}.npy"

    if not file_path.exists():
        raise FileNotFoundError(f"Particles file not found: {file_path}")
    return file_path


def resolve_mjdata_path(
    source: ParticleSource,
    *,
    which: Literal["latest", "iter"] = "latest",
    iteration: int | None = None,
) -> Path | None:
    if which == "latest":
        latest_path = source.particles_dir / "mjdata_latest.npz"
        if latest_path.exists():
            return latest_path
        latest_iter_path = source.particles_dir / "mjdata_latest_iter.txt"
        if latest_iter_path.exists():
            try:
                iter_val = int(latest_iter_path.read_text(encoding="utf-8").strip())
                candidate = source.particles_dir / f"mjdata_iter_{iter_val:05d}.npz"
                if candidate.exists():
                    return candidate
            except ValueError:
                pass
        candidates = sorted(source.particles_dir.glob("mjdata_iter_*.npz"))
        if candidates:
            return candidates[-1]
        return None
    if iteration is None:
        raise ValueError("iteration must be provided when which='iter'")
    candidate = source.particles_dir / f"mjdata_iter_{iteration:05d}.npz"
    return candidate if candidate.exists() else None


def load_costs(
    source: ParticleSource,
    *,
    which: Literal["latest", "iter"] = "latest",
    iteration: int | None = None,
) -> np.ndarray:
    """Load rollout costs corresponding to the saved particles."""
    if which == "latest":
        file_path = source.particles_dir / "costs_latest.npy"
        if not file_path.exists():
            candidates = sorted(source.particles_dir.glob("costs_iter_*.npy"))
            if not candidates:
                raise FileNotFoundError(f"Costs file not found: {file_path}")
            file_path = candidates[-1]
            print(f"costs_latest.npy missing; using {file_path.name}")
    else:
        if iteration is None:
            raise ValueError("iteration must be provided when which='iter'")
        file_path = source.particles_dir / f"costs_iter_{iteration:05d}.npy"

    if not file_path.exists():
        raise FileNotFoundError(f"Costs file not found: {file_path}")
    return np.load(file_path)


def compute_costs(
    optimizer: TrajectoryOptimizer,
    particles: np.ndarray,
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    """Compute rollout costs for each particle (knot sequence)."""
    if particles.ndim != 3:
        raise ValueError(f"Expected particles with shape (N,K,nu), got {particles.shape}")

    batch_size = batch_size or getattr(optimizer, "max_eval", None) or particles.shape[0]
    batch_size = max(int(batch_size), 1)

    ctrl = optimizer.controller
    task = ctrl.task

    optimizer.reset_mjx_data()
    state = optimizer.mjx_data

    n_particles = particles.shape[0]
    costs_out = np.empty((n_particles,), dtype=np.float32)

    for start in range(0, n_particles, batch_size):
        end = min(start + batch_size, n_particles)
        knots_batch = jnp.asarray(particles[start:end])
        controls_batch = optimizer.knots2ctrls(knots_batch)
        _, rollouts = ctrl.eval_rollouts(task.model, state, controls_batch, knots_batch)
        costs_batch = jax.device_get(jnp.sum(rollouts.costs, axis=-1))
        costs_out[start:end] = np.asarray(costs_batch, dtype=np.float32)

    return costs_out


def _read_base_height(source: ParticleSource, iteration: int) -> float | None:
    term_path = source.experiment_dir / "stats" / "terminal_costs.csv"
    if not term_path.exists():
        return None
    try:
        with term_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "base_height" not in reader.fieldnames:
                return None
            height_by_iter: dict[int, float] = {}
            for row in reader:
                try:
                    iter_val = int(row.get("iteration", "-1"))
                    height_by_iter[iter_val] = float(row["base_height"])
                except (TypeError, ValueError):
                    continue
    except OSError:
        return None
    if iteration in height_by_iter:
        return height_by_iter[iteration]
    if iteration > 0 and (iteration - 1) in height_by_iter:
        return height_by_iter[iteration - 1]
    return None


def _parse_iter_from_stem(stem: str, prefix: str) -> int | None:
    if not stem.startswith(prefix):
        return None
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return None


def select_knots(
    particles: np.ndarray,
    *,
    particle_index: int = 0,
) -> jnp.ndarray:
    """Select one particle (knot sequence) from a `(N, K, nu)` particles array."""
    if particles.ndim != 3:
        raise ValueError(f"Expected particles with shape (N,K,nu), got {particles.shape}")
    if not (0 <= particle_index < particles.shape[0]):
        raise IndexError(f"particle_index {particle_index} out of range [0, {particles.shape[0]})")
    return jnp.asarray(particles[particle_index])


def select_knots_by_rank(
    particles: np.ndarray,
    costs: np.ndarray,
    *,
    rank: int = 0,
) -> tuple[jnp.ndarray, int, float]:
    """
    Select knots by sorting particles by cost (ascending) and taking `rank`.

    Returns `(knots, particle_index, cost)`.
    """
    if particles.ndim != 3:
        raise ValueError(f"Expected particles with shape (N,K,nu), got {particles.shape}")
    costs = np.asarray(costs).reshape(-1)
    if costs.shape[0] != particles.shape[0]:
        raise ValueError(f"Costs length {costs.shape[0]} != num particles {particles.shape[0]}")
    if not (0 <= rank < particles.shape[0]):
        raise IndexError(f"rank {rank} out of range [0, {particles.shape[0]})")

    order = np.argsort(costs)
    particle_index = int(order[rank])
    cost = float(costs[particle_index])
    return jnp.asarray(particles[particle_index]), particle_index, cost


def particle_to_gif(
    optimizer: TrajectoryOptimizer,
    *,
    experiment_dir: str | Path | None = None,
    algorithm: str | None = None,
    iteration: int | None = None,
    rank: int | None = None,
    show_reference: bool = True,
    out_path: str | Path | None = None,
    playback_dt: float | None = None,
    frame_stride: int = 1,
    playback_speed: float = 1.0,
) -> Path:
    """
    Load a particle (knots), simulate it in MuJoCo, and write a GIF.
    """
    which: Literal["latest", "iter"] = "iter" if iteration is not None else "latest"
    source = resolve_experiment_dir(experiment_dir, algorithm=algorithm)
    use_consensus = rank is None
    iter_for_stats: int | None = iteration
    mjdata_path = resolve_mjdata_path(source, which=which, iteration=iteration)
    if use_consensus:
        if which == "latest":
            consensus_path = source.particles_dir / "consensus_latest.npy"
            if not consensus_path.exists():
                candidates = sorted(source.particles_dir.glob("consensus_iter_*.npy"))
                if not candidates:
                    raise FileNotFoundError(f"Consensus files not found under {source.particles_dir}")
                consensus_path = candidates[-1]
            if iter_for_stats is None:
                iter_for_stats = _parse_iter_from_stem(consensus_path.stem, "consensus_iter_")
        else:
            if iteration is None:
                raise ValueError("iteration must be provided when which='iter'")
            consensus_path = source.particles_dir / f"consensus_iter_{iteration:05d}.npy"
            if not consensus_path.exists():
                raise FileNotFoundError(f"Consensus file not found: {consensus_path}")
        knots = jnp.asarray(np.load(consensus_path))
        particle_index = None
        cost = None
    else:
        file_path = resolve_particles_path(source, which=which, iteration=iteration)
        if iter_for_stats is None:
            iter_for_stats = _parse_iter_from_stem(file_path.stem, "particles_iter_")
        particles = np.load(file_path)
        try:
            costs = load_costs(source, which=which, iteration=iteration)
        except FileNotFoundError:
            costs = compute_costs(optimizer, particles)
            if which == "latest":
                np.save(source.particles_dir / "costs_latest.npy", costs)
            else:
                if iteration is None:
                    raise ValueError("iteration must be provided when which='iter'")
                np.save(source.particles_dir / f"costs_iter_{iteration:05d}.npy", costs)
        knots, particle_index, cost = select_knots_by_rank(particles, costs, rank=rank)
    if mjdata_path is None and iter_for_stats is not None:
        mjdata_path = resolve_mjdata_path(source, which="iter", iteration=iter_for_stats)
    controls = optimizer.knots2ctrls(knots)

    base_playback_dt = playback_dt
    if base_playback_dt is None:
        base_playback_dt = float(optimizer.mj_model.opt.timestep)
    # Shared GIF playback settings (keep replay and persistence aligned).
    effective_playback_dt, frame_stride, playback_suffix, fps = compute_gif_playback_settings(
        base_playback_dt,
        playback_speed=playback_speed,
        frame_stride=frame_stride,
    )

    if out_path is None:
        if which == "latest":
            latest_iter_path = source.particles_dir / "particles_latest_iter.txt"
            if latest_iter_path.exists():
                try:
                    iter_val = int(latest_iter_path.read_text(encoding="utf-8").strip())
                    tag = f"iter_{iter_val:05d}"
                    if iter_for_stats is None:
                        iter_for_stats = iter_val
                except ValueError:
                    tag = "latest"
            elif not use_consensus and file_path.name.startswith("particles_iter_"):
                iter_tag = file_path.stem.split("_")[-1]
                tag = f"iter_{iter_tag}"
            else:
                tag = "latest"
        else:
            tag = f"iter_{iteration:05d}"
        if use_consensus:
            base_height = None
            if iter_for_stats is not None:
                base_height = _read_base_height(source, iter_for_stats)
            base_height_suffix = "_base_height_unknown"
            if base_height is not None:
                base_height_suffix = f"_base_height_{base_height:.4f}"
            out_path = source.experiment_dir / "gifs" / f"consensus_{tag}{playback_suffix}{base_height_suffix}.gif"
        else:
            base_height = None
            if iter_for_stats is not None:
                base_height = _read_base_height(source, iter_for_stats)
            base_height_suffix = "_base_height_unknown"
            if base_height is not None:
                base_height_suffix = f"_base_height_{base_height:.4f}"
            out_path = (
                source.experiment_dir
                / "gifs"
                / f"particle_{tag}_rank_{rank:04d}_idx_{particle_index:04d}_cost_{cost:.6g}{playback_suffix}{base_height_suffix}.gif"
            )
    frame_indices = build_stride_frame_indices(controls.shape[0], frame_stride)
    optimizer.visuals.savegif(
        np.asarray(controls),
        show_reference=show_reference,
        extra_fname="particle_replay",
        out_path=out_path,
        playback_dt=effective_playback_dt,
        frame_stride=frame_stride,
        frame_indices=frame_indices,
        mjdata_path=mjdata_path,
    )
    return Path(out_path)


__all__ = [
    "ParticleSource",
    "resolve_experiment_dir",
    "resolve_particles_path",
    "load_particles",
    "load_costs",
    "compute_costs",
    "select_knots",
    "select_knots_by_rank",
    "particle_to_gif",
]


def _setting(settings: dict[str, object], *keys: str, default):
    for key in keys:
        if key in settings:
            return settings[key]
    return default


def _build_optimizer_from_compare(
    task_name: str,
    *,
    algorithm_name: str | None = None,
) -> TrajectoryOptimizer:
    config_path = Path("config/compare_config.json")
    config = load_compare_config(config_path)
    settings = config.settings_for_algorithm(algorithm_name)
    import mujoco
    from hydrax.task_base import Task
    from utils.mjx_utils import apply_mjx_settings
    from tasks.double_cartpole import DoubleCartPoleUnconstrained
    from tasks.humanoid import Humanoid

    if task_name == "humanoid":
        task: Task = Humanoid(
            reference_filename="DefaultDatasets/mocap/UnitreeG1/balance.npz",
            start_frame=0,
        )
        apply_mjx_settings(task, config.mjx_integration_time)
        mj_model = task.mj_model
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = task.start_config
    elif task_name == "double_cartpole":
        task = DoubleCartPoleUnconstrained()
        mj_model = task.mj_model
        mj_data = mujoco.MjData(mj_model)
    else:
        raise ValueError("task must be one of: humanoid, double_cartpole")

    from algs.cbo_x import CBOx
    try:
        from pcbo.cbox_polarized import CBOxPolarized
    except ImportError:
        CBOxPolarized = None

    _default = "CBOxPolarized" if CBOxPolarized is not None else "CBOx"
    controller_name = (
        algorithm_name
        if algorithm_name in {"CBOx", "CBOxPolarized"}
        else _default
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
        spline_type="zero",
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
    persist_particles_every = _setting(
        settings, "persist_particles_every", default=config.persist_particles_every
    )
    persist_particles_latest = _setting(
        settings, "persist_particles_latest", default=config.persist_particles_latest
    )
    if persist_particles_every is not None:
        ctrl.persist_particles_every = persist_particles_every
    if persist_particles_latest is not None:
        ctrl.persist_particles_latest = persist_particles_latest
    return TrajectoryOptimizer(
        controller_name,
        ctrl,
        mj_model,
        mj_data,
        max_eval=config.max_eval,
        config_path=config_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replay saved CBOx particles to a MuJoCo GIF.")
    parser.add_argument("--experiment-dir", type=str, default=None, help="Path under tmp/experiments; defaults to latest.")
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Subdir name under compare_<timestamp>/ (e.g. CBOx).",
    )
    parser.add_argument("--iter", type=int, default=None, help="Iteration to load (e.g. 500).")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Replay the k-th best particle (0 = best). If omitted, use consensus.",
    )
    parser.add_argument("--task", choices=["humanoid", "double_cartpole"], default="humanoid", help="Task to simulate.")
    parser.add_argument("--out", type=str, default=None, help="Output GIF path; defaults under the experiment dir.")
    parser.add_argument("--no-reference", action="store_true", help="Disable reference ghost rendering.")
    parser.add_argument(
        "--playback-dt",
        type=float,
        default=None,
        help="Playback timestep for GIF FPS (defaults to MJX integration dt).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Render every Nth simulation step to speed up playback (auto-increased if GIF fps caps).",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=8.0,
        help="Playback speed multiplier (e.g. 4 for 4x faster).",
    )
    args = parser.parse_args()

    optimizer = _build_optimizer_from_compare(args.task, algorithm_name=args.algorithm)
    gif_path = particle_to_gif(
        optimizer,
        experiment_dir=args.experiment_dir,
        algorithm=args.algorithm,
        iteration=args.iter,
        rank=args.rank,
        show_reference=not args.no_reference,
        out_path=args.out,
        playback_dt=args.playback_dt,
        frame_stride=args.frame_stride,
        playback_speed=args.playback_speed,
    )
    print(gif_path)
