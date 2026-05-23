"""Run DOOM's RL contact policy on Unitree Go2 in MuJoCo (trot in place).

Loads the Go2 MJCF from this repo's `models/go2/scene.xml`, drives the robot
with the TorchScript policy `contact_policy_stanceSeperate.pt`, and reproduces
the 82-dim observation vector + action processing used by
`RLQuadrupedLocomotionContactController`. Gait is fixed to trot with
`command_duration=0.35s`, no stride, no heading -- so the robot trots in place.
"""

import argparse
import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_XML = REPO_ROOT / "models" / "go2" / "scene.xml"
POLICY_FILE = str(REPO_ROOT / "models" / "go2" / "policies" / "contact_policy_stanceSeperate.pt")

SIM_DT = 0.005
DECIMATION = 4
ACTION_SCALE = 0.35
COMMAND_DURATION = 0.35
HORIZON = 16
OBS_HORIZON = 2

MJ_TO_ISAAC = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])
ISAAC_TO_MJ = np.array([1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10])

DEFAULT_Q_ISAAC = np.array(
    [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
    dtype=np.float32,
)
DEFAULT_Q_MJ = DEFAULT_Q_ISAAC[ISAAC_TO_MJ]

# Feet target offsets in base frame, ordered [FL, FR, RL, RR].
DEFAULT_OFFSET = np.array(
    [
        (0.1934, 0.1465, 0.0),
        (0.1934, -0.1465, 0.0),
        (-0.1934, 0.1465, 0.0),
        (-0.1934, -0.1465, 0.0),
    ],
    dtype=np.float32,
)
FEET_TARGET_Z = 0.0

# Trot contact pattern, 2 phases x 4 feet in [FL, FR, RL, RR] order.
TROT_PATTERN = np.array(
    [
        [False, True, True, False],
        [True, False, False, True],
    ],
    dtype=np.float32,
)

FOOT_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# Joint names in the order DOOM treats as "MuJoCo" (matches Go2's <actuator> block).
# We DO NOT assume this equals the order of data.qpos[7:19]; instead we look up qpos/qvel
# indices for each joint by name. The actuators happen to be declared in this same order,
# so data.ctrl[i] directly corresponds to MJ_JOINT_NAMES[i].
MJ_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by the inverse of quaternion q (w,x,y,z).

    Works with v of shape (3,) or (..., 3).
    """
    w = q_wxyz[0]
    xyz = q_wxyz[1:4]
    v = np.asarray(v, dtype=np.float32)
    # Using the identity v' = (2 w^2 - 1) v - 2 w (xyz x v) + 2 xyz (xyz . v)
    cross = np.cross(xyz, v)
    dot = np.einsum("...i,i->...", v, xyz)
    return (2.0 * w * w - 1.0) * v - 2.0 * w * cross + 2.0 * dot[..., None] * xyz


def build_future_feet_w(base_xy: np.ndarray) -> np.ndarray:
    """Constant-along-horizon foot targets at default offsets from `base_xy`, z=0.1."""
    base = np.array([base_xy[0], base_xy[1], FEET_TARGET_Z], dtype=np.float32)
    per_foot = base + DEFAULT_OFFSET  # shape (4, 3)
    return np.broadcast_to(per_foot[:, None, :], (4, HORIZON, 3)).copy()


def build_observation(
    data: mujoco.MjData,
    foot_body_ids: list[int],
    qpos_addr: np.ndarray,
    qvel_addr: np.ndarray,
    future_feet_w: np.ndarray,
    current_goal_idx: int,
    current_contact_plan: np.ndarray,
    time_left: float,
    last_action_isaac: np.ndarray,
) -> np.ndarray:
    base_quat = np.array(data.qpos[3:7], dtype=np.float32)
    base_pos = np.array(data.qpos[0:3], dtype=np.float32)

    lin_vel_w = np.array(data.qvel[0:3], dtype=np.float32)
    lin_vel_b = quat_rotate_inverse(base_quat, lin_vel_w)
    # MuJoCo free joints store angular velocity already in the body frame.
    ang_vel_b = np.array(data.qvel[3:6], dtype=np.float32)
    proj_grav = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))

    # contact_locations_b: future feet in [current_goal_idx, current_goal_idx + OBS_HORIZON) -> base frame.
    future_slice = future_feet_w[:, current_goal_idx : current_goal_idx + OBS_HORIZON, :]
    rel = future_slice - base_pos  # (4, OBS_HORIZON, 3)
    contact_locations_b = quat_rotate_inverse(base_quat, rel).reshape(-1)

    contact_plan = current_contact_plan.reshape(-1).astype(np.float32)

    qpos_mj = data.qpos[qpos_addr].astype(np.float32)
    qvel_mj = data.qvel[qvel_addr].astype(np.float32)
    joint_pos_rel = qpos_mj[MJ_TO_ISAAC] - DEFAULT_Q_ISAAC
    joint_vel = qvel_mj[MJ_TO_ISAAC]

    foot_pos_w = np.stack([data.xpos[bid] for bid in foot_body_ids], axis=0).astype(np.float32)
    ee_pos_rel_b = np.linalg.norm(future_feet_w[:, current_goal_idx, :] - foot_pos_w, axis=1).astype(np.float32)

    obs = np.concatenate(
        [
            lin_vel_b,
            ang_vel_b,
            proj_grav,
            contact_locations_b,
            np.array([time_left], dtype=np.float32),
            contact_plan,
            joint_pos_rel,
            joint_vel,
            ee_pos_rel_b,
            last_action_isaac,
        ]
    ).astype(np.float32)
    assert obs.shape == (82,), f"obs shape {obs.shape} != (82,)"
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=POLICY_FILE, help="Path to TorchScript policy.")
    parser.add_argument("--no-viewer", action="store_true", help="Run headless without launching the viewer.")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0 = run forever).")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)

    # Index arrays so we can read joint pos/vel in DOOM's MuJoCo order regardless of
    # the actual joint-tree order MuJoCo uses internally.
    qpos_addr = np.array([model.jnt_qposadr[model.joint(name).id] for name in MJ_JOINT_NAMES])
    qvel_addr = np.array([model.jnt_dofadr[model.joint(name).id] for name in MJ_JOINT_NAMES])

    data.qpos[0:3] = [0.0, 0.0, 0.4]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_addr] = DEFAULT_Q_MJ
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    foot_body_ids = [model.body(name).id for name in FOOT_BODY_NAMES]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = torch.jit.load(args.policy, map_location=device).eval()

    last_action_isaac = np.zeros(12, dtype=np.float32)
    joint_pos_targets_mj = DEFAULT_Q_MJ.copy()

    base_xy0 = np.array(data.qpos[0:2], dtype=np.float32)
    future_feet_w = build_future_feet_w(base_xy0)
    current_contact_plan = TROT_PATTERN.copy()
    current_goal_idx = 0
    goal_completion_counter = 0
    command_start = time.time()

    ctrlrange_lo = model.actuator_ctrlrange[:, 0]
    ctrlrange_hi = model.actuator_ctrlrange[:, 1]

    step_count = 0
    t_start = time.time()

    def loop_body():
        nonlocal current_contact_plan, current_goal_idx, goal_completion_counter
        nonlocal command_start, joint_pos_targets_mj, last_action_isaac, step_count

        loop_t0 = time.time()
        now = loop_t0

        elapsed = now - command_start
        time_left = max(0.0, COMMAND_DURATION - elapsed)
        if elapsed >= COMMAND_DURATION:
            goal_completion_counter += 1
            if goal_completion_counter % 2 != 0:
                current_goal_idx += 1
            current_contact_plan = current_contact_plan[[1, 0]]
            command_start = now
            if current_goal_idx + OBS_HORIZON > HORIZON:
                current_goal_idx = 0  # safe: feet targets are constant in trot-in-place
            time_left = COMMAND_DURATION

        if step_count % DECIMATION == 0:
            obs = build_observation(
                data,
                foot_body_ids,
                qpos_addr,
                qvel_addr,
                future_feet_w,
                current_goal_idx,
                current_contact_plan,
                time_left,
                last_action_isaac,
            )
            with torch.no_grad():
                raw = policy(torch.from_numpy(obs).to(device).unsqueeze(0))
            raw_isaac = raw.squeeze(0).cpu().numpy().astype(np.float32)
            last_action_isaac[:] = raw_isaac
            q_target_isaac = raw_isaac * ACTION_SCALE + DEFAULT_Q_ISAAC
            joint_pos_targets_mj = q_target_isaac[ISAAC_TO_MJ]

        np.clip(joint_pos_targets_mj, ctrlrange_lo, ctrlrange_hi, out=joint_pos_targets_mj)
        data.ctrl[:] = joint_pos_targets_mj

        mujoco.mj_step(model, data)
        step_count += 1

        sleep_for = SIM_DT - (time.time() - loop_t0)
        if sleep_for > 0:
            time.sleep(sleep_for)

    if args.no_viewer:
        while True:
            loop_body()
            if args.duration > 0 and (time.time() - t_start) >= args.duration:
                break
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body("base_link").id
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -20.0
            while viewer.is_running():
                loop_body()
                viewer.sync()
                if args.duration > 0 and (time.time() - t_start) >= args.duration:
                    break


if __name__ == "__main__":
    main()
