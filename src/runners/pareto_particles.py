from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np

from runners.particle_replay import resolve_experiment_dir, resolve_particles_path, resolve_mjdata_path
from trajopt.helpers import _capture_mj_data_state, apply_mj_data_state, load_mj_data_state
from runners.particle_replay import _build_optimizer_from_compare
from utils.humanoid_standing import support_polygon_cost


@dataclass
class _StateView:
    qpos: jnp.ndarray
    qvel: jnp.ndarray
    time: jnp.ndarray
    sensordata: jnp.ndarray
    userdata: jnp.ndarray


def _evaluate_objectives(optimizer, controls: np.ndarray, base_state: dict) -> tuple[float, float, float]:
    mj_model = optimizer.mj_model
    task = optimizer.controller.task

    mj_data = mujoco.MjData(mj_model)
    apply_mj_data_state(mj_data, base_state)
    mujoco.mj_forward(mj_model, mj_data)

    for u in controls:
        mj_data.ctrl[:] = u
        mujoco.mj_step(mj_model, mj_data)

    state = _StateView(
        qpos=jnp.asarray(mj_data.qpos),
        qvel=jnp.asarray(mj_data.qvel),
        time=jnp.asarray(mj_data.time),
        sensordata=jnp.asarray(mj_data.sensordata),
        userdata=jnp.asarray(mj_data.userdata) if mj_data.userdata.size else jnp.asarray([]),
    )

    base_pos_cost, base_ori_cost, ori_weight, weighted_sum, _ = task.base_cost_terms(state)
    q_ref = task._get_reference_configuration(state.time)
    base_height_cost = jnp.square(state.qpos[2] - q_ref[2])
    left_pos, right_pos = task._get_foot_positions(state)
    support_cost = support_polygon_cost(
        state.qpos[:2],
        jnp.stack([left_pos, right_pos]),
        support_margin=task.stand_support_margin,
    )

    return float(weighted_sum), float(base_height_cost), float(support_cost)


def _pareto_mask(costs: np.ndarray) -> np.ndarray:
    n = costs.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominates = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        is_pareto[dominates] = False
    return is_pareto


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Pareto set for saved particles.")
    parser.add_argument("--experiment-dir", type=str, default=None, help="Path under tmp/experiments; defaults to latest.")
    parser.add_argument("--algorithm", type=str, default=None, help="Algorithm subdir under compare_<timestamp>/ (e.g. CBOx).")
    parser.add_argument("--iter", type=int, default=None, help="Iteration to load (e.g. 500).")
    parser.add_argument("--task", choices=["humanoid"], default="humanoid", help="Task to simulate.")
    args = parser.parse_args()

    if args.task != "humanoid":
        raise ValueError("Only humanoid is supported for the requested objectives.")

    optimizer = _build_optimizer_from_compare(args.task)
    which = "iter" if args.iter is not None else "latest"
    source = resolve_experiment_dir(args.experiment_dir, algorithm=args.algorithm)
    particles_path = resolve_particles_path(source, which=which, iteration=args.iter)
    particles = np.load(particles_path)

    mjdata_path = resolve_mjdata_path(source, which=which, iteration=args.iter)
    if mjdata_path is not None:
        base_state = load_mj_data_state(mjdata_path)
    else:
        base_state = _capture_mj_data_state(optimizer.mj_data)

    objectives = np.zeros((particles.shape[0], 3), dtype=np.float64)
    for i, knots in enumerate(particles):
        controls = np.asarray(optimizer.knots2ctrls(knots))
        objectives[i] = _evaluate_objectives(optimizer, controls, base_state)

    pareto_mask = _pareto_mask(objectives)
    pareto_count = int(pareto_mask.sum())
    total = int(objectives.shape[0])
    percent = 100.0 * pareto_count / max(total, 1)
    print(f"pareto_count={pareto_count} total={total} percent={percent:.2f}%")


if __name__ == "__main__":
    main()
