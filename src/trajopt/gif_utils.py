from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from utils.log_utils import log_line


def _resolve_last_output_path(optimizer):
    artifacts = getattr(optimizer, "artifacts", None)
    if artifacts is not None and hasattr(artifacts, "get_last_output_path"):
        return artifacts.get_last_output_path()
    if hasattr(optimizer, "_get_last_output_path"):
        return optimizer._get_last_output_path()
    return None


def _base_height_suffix(optimizer, iteration: int) -> str:
    base_height = optimizer._read_terminal_base_height(iteration)
    if base_height is None:
        return "_base_height_unknown"
    return f"_base_height_{base_height:.4f}"


def _save_gif(
    optimizer,
    *,
    controls: np.ndarray,
    out_path,
    extra_fname: str,
    playback_dt: float,
    frame_stride: int,
    success_msg: str,
    error_msg: str,
) -> None:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gif_path = optimizer.visuals.savegif(
            np.asarray(controls),
            out_path=out_path,
            extra_fname=extra_fname,
            playback_dt=playback_dt,
            frame_stride=frame_stride,
        )
        if gif_path is not None:
            log_line(success_msg.format(path=gif_path), _resolve_last_output_path(optimizer))
    except Exception as exc:
        log_line(error_msg.format(exc=exc), _resolve_last_output_path(optimizer))


def save_ranked_particle_gifs(
    optimizer,
    *,
    particles_np: np.ndarray,
    rollout_costs: np.ndarray,
    iteration: int,
    playback_suffix: str,
    playback_dt: float,
    frame_stride: int,
) -> None:
    if particles_np.ndim != 3 or particles_np.shape[0] == 0:
        return

    rank_count = int(
        getattr(optimizer.controller, "persist_particle_gif_count", 1) or 1
    )
    rank_count = max(1, min(rank_count, particles_np.shape[0]))
    rank_order = np.argsort(rollout_costs)

    base_height_suffix = _base_height_suffix(optimizer, iteration)

    for rank, best_index in enumerate(rank_order[:rank_count]):
        try:
            best_knots = particles_np[best_index]
            controls = optimizer.knots2ctrls(best_knots)
            best_cost = float(rollout_costs[best_index])
            out_path = (
                optimizer.run_dir
                / "gifs"
                / "particles"
                / f"particle_iter_{iteration:05d}_rank_{rank:04d}_idx_{best_index:04d}_cost_{best_cost:.6g}"
                f"{playback_suffix}{base_height_suffix}.gif"
            )
            _save_gif(
                optimizer,
                controls=controls,
                out_path=out_path,
                extra_fname="particles",
                playback_dt=playback_dt,
                frame_stride=frame_stride,
                success_msg="saved particles gif: {path}",
                error_msg="particles gif render failed: {exc}",
            )
        except Exception as exc:
            log_line(
                f"particles gif render failed: {exc}",
                _resolve_last_output_path(optimizer),
            )


def save_consensus_gif(
    optimizer,
    *,
    mean_knots,
    iteration: int,
    playback_suffix: str,
    playback_dt: float,
    frame_stride: int,
) -> None:
    try:
        consensus_np = np.asarray(jax.device_get(mean_knots))
    except Exception:
        consensus_np = np.asarray(mean_knots)
    if consensus_np.ndim != 2:
        return
    try:
        controls = optimizer.knots2ctrls(consensus_np)
        consensus_cost_suffix = "_cost_unknown"
        try:
            ctrl = optimizer.controller
            task = ctrl.task
            controls_batch = controls[None, ...]
            knots_batch = consensus_np[None, ...]
            _, rollouts = ctrl.eval_rollouts(
                task.model, optimizer.mjx_data, controls_batch, knots_batch
            )
            consensus_cost = float(np.asarray(jnp.sum(rollouts.costs, axis=-1)[0]))
            consensus_cost_suffix = f"_cost_{consensus_cost:.6g}"
        except Exception:
            pass
        base_height_suffix = _base_height_suffix(optimizer, iteration)
        out_path = (
            optimizer.run_dir
            / "gifs"
            / "consensus"
            / f"consensus_iter_{iteration:05d}{consensus_cost_suffix}{playback_suffix}{base_height_suffix}.gif"
        )
        _save_gif(
            optimizer,
            controls=controls,
            out_path=out_path,
            extra_fname="consensus",
            playback_dt=playback_dt,
            frame_stride=frame_stride,
            success_msg="saved consensus gif: {path}",
            error_msg="consensus gif render failed: {exc}",
        )
    except Exception as exc:
        log_line(
            f"consensus gif render failed: {exc}",
            _resolve_last_output_path(optimizer),
        )
