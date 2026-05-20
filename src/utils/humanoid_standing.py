from __future__ import annotations

import jax.numpy as jnp
from mujoco.mjx._src.math import quat_sub


def is_standing_straight(
    qpos: jnp.ndarray,
    q_ref: jnp.ndarray,
    *,
    base_pos_cost_tol: float,
    base_ori_cost_tol: float,
    min_base_height: float | None = None,
) -> jnp.ndarray:
    """Return True if base position/orientation are within tolerances."""
    base_pos_err = qpos[:3] - q_ref[:3]
    base_ori_err = quat_sub(qpos[3:7], q_ref[3:7])
    base_pos_cost = jnp.sum(jnp.square(base_pos_err))
    base_ori_cost = jnp.sum(jnp.square(base_ori_err))

    pos_ok = base_pos_cost <= base_pos_cost_tol
    ori_ok = base_ori_cost <= base_ori_cost_tol
    if min_base_height is None:
        height_ok = jnp.array(True)
    else:
        height_ok = qpos[2] >= min_base_height
    return pos_ok & ori_ok & height_ok


def is_standing_stable(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    q_ref: jnp.ndarray,
    *,
    base_pos_cost_tol: float,
    base_ori_cost_tol: float,
    min_base_height: float | None = None,
    max_base_lin_vel: float | None = None,
    max_base_ang_vel: float | None = None,
    foot_positions: jnp.ndarray | None = None,
    support_margin: float = 0.0,
    joint_range: jnp.ndarray | None = None,
    max_joint_limit_violation: float | None = None,
) -> jnp.ndarray:
    """Return True if base is upright, slow, and supported."""
    upright_ok = is_standing_straight(
        qpos,
        q_ref,
        base_pos_cost_tol=base_pos_cost_tol,
        base_ori_cost_tol=base_ori_cost_tol,
        min_base_height=min_base_height,
    )

    if max_base_lin_vel is None:
        lin_ok = jnp.array(True)
    else:
        lin_speed = jnp.linalg.norm(qvel[:3])
        lin_ok = lin_speed <= max_base_lin_vel

    if max_base_ang_vel is None:
        ang_ok = jnp.array(True)
    else:
        ang_speed = jnp.linalg.norm(qvel[3:6])
        ang_ok = ang_speed <= max_base_ang_vel

    if foot_positions is None: # foot position should be passed from sensor
        support_ok = jnp.array(True)
    else:
        base_xy = qpos[:2]
        foot_xy = foot_positions[:, :2]
        min_xy = jnp.min(foot_xy, axis=0) - support_margin
        max_xy = jnp.max(foot_xy, axis=0) + support_margin
        support_ok = jnp.all(base_xy >= min_xy) & jnp.all(base_xy <= max_xy)

    if joint_range is None or max_joint_limit_violation is None:
        joint_ok = jnp.array(True)
    else:
        if joint_range.shape[0] != qpos.shape[0]:
            joint_ok = jnp.array(True)
        else:
            lower = joint_range[:, 0]
            upper = joint_range[:, 1]
            violation = jnp.maximum(qpos - upper, 0.0) + jnp.maximum(lower - qpos, 0.0)
            joint_ok = jnp.max(violation) <= max_joint_limit_violation

    return upright_ok & lin_ok & ang_ok & support_ok & joint_ok


def support_polygon_cost(
    base_xy: jnp.ndarray,
    foot_positions: jnp.ndarray,
    *,
    support_margin: float = 0.0,
) -> jnp.ndarray:
    """Distance-based penalty for base outside the foot support box."""
    if foot_positions is None:
        return jnp.array(0.0)
    foot_xy = foot_positions[:, :2]
    min_xy = jnp.min(foot_xy, axis=0) - support_margin
    max_xy = jnp.max(foot_xy, axis=0) + support_margin
    below = jnp.maximum(min_xy - base_xy, 0.0)
    above = jnp.maximum(base_xy - max_xy, 0.0)
    return jnp.sum(below + above)
