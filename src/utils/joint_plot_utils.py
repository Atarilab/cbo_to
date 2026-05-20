from __future__ import annotations

from typing import Iterable

import numpy as np


LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)


def _unique_sorted(values: Iterable[int]) -> list[int]:
    return sorted({int(v) for v in values})


def get_leg_qpos_indices(mj_model) -> list[int]:
    indices = []
    for name in LEG_JOINT_NAMES:
        try:
            jid = mj_model.joint(name).id
        except Exception:
            continue
        indices.append(int(mj_model.jnt_qposadr[jid]))
    return _unique_sorted(indices)


def get_leg_qvel_indices(mj_model) -> list[int]:
    indices = []
    for name in LEG_JOINT_NAMES:
        try:
            jid = mj_model.joint(name).id
        except Exception:
            continue
        indices.append(int(mj_model.jnt_dofadr[jid]))
    return _unique_sorted(indices)


def get_leg_actuator_indices(mj_model) -> list[int]:
    indices = []
    for name in LEG_JOINT_NAMES:
        try:
            aid = mj_model.actuator(name).id
        except Exception:
            continue
        indices.append(int(aid))
    return _unique_sorted(indices)
