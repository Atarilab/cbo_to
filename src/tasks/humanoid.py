from typing import Dict

import jax
import jax.numpy as jnp
import utils.mujoco_env as mujoco_env  # sets MUJOCO_GL early for headless runs
import mujoco
import numpy as np
from huggingface_hub import hf_hub_download
from mujoco import mjx
from mujoco.mjx._src.math import quat_sub

from hydrax import ROOT
from hydrax.task_base import Task
from utils.humanoid_standing import is_standing_stable, support_polygon_cost
from loss.se3_twist import SE3TwistCalculator
from utils.base_weight_scheduler import BaseOrientationWeightScheduler
from utils.terminal_cost_controller import TerminalCostController
# This task is modified from Hydrax, added a starting frame option
# https://github.com/vincekurtz/hydrax/tree/main/hydrax/tasks


class Humanoid(Task):
    """The Unitree G1 humanoid tracks a reference from motion capture.

    Retargeted motion capture data comes from the LocoMuJoCo dataset:
    https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.
    """

    def __init__(
        self,
        reference_filename: str = "Lafan1/mocap/UnitreeG1/walk1_subject1.npz",
        model_xml_path: str | None = None,
        terminal_full_config_cost_weight: float = 0.0,
        terminal_base_se3_weight: float = 50.0,
        terminal_foot_position_weight: float = 0.0,
        base_ori_weight_start: float = 1.0,
        base_ori_weight_end: float = 1.0,
        base_weight_scheduler: BaseOrientationWeightScheduler | None = None,
        terminal_cost_controller: TerminalCostController | None = None,
        stand_base_pos_cost_tol: float = 0.01,
        stand_base_ori_cost_tol: float = 0.01,
        stand_min_base_height: float | None = None,
        terminal_fall_penalty: float = 1000.0,
        stand_max_base_lin_vel: float | None = None,
        stand_max_base_ang_vel: float | None = None,
        stand_support_margin: float = 0.0,
        stand_joint_limit_violation: float | None = None,
        use_se3_twist: bool = False,
        start_frame: int = 100,
        flag_zero_running_cost: bool = False,
    ) -> None:
        """Load the MuJoCo model and set task parameters.

        The list of available reference files can be found at
        https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.
        """
        xml_path = model_xml_path or (ROOT + "/models/g1/scene_23dof.xml")
        mj_model = mujoco.MjModel.from_xml_path(xml_path)

        self.terminal_full_config_cost_weight = terminal_full_config_cost_weight
        self.terminal_base_se3_weight = terminal_base_se3_weight
        self.terminal_foot_position_weight = terminal_foot_position_weight
        self.base_ori_weight_start = base_ori_weight_start
        self.base_ori_weight_end = base_ori_weight_end
        self.base_ori_weight = base_ori_weight_start
        self.base_weight_scheduler = base_weight_scheduler
        self.terminal_cost_controller = terminal_cost_controller
        self.use_se3_twist = use_se3_twist
        self._se3_twist_calc = SE3TwistCalculator() if use_se3_twist else None
        self.stand_base_pos_cost_tol = stand_base_pos_cost_tol
        self.stand_base_ori_cost_tol = stand_base_ori_cost_tol
        self.stand_min_base_height = stand_min_base_height
        self.terminal_fall_penalty = terminal_fall_penalty
        self.stand_max_base_lin_vel = stand_max_base_lin_vel
        self.stand_max_base_ang_vel = stand_max_base_ang_vel
        self.stand_support_margin = stand_support_margin
        self.stand_joint_limit_violation = stand_joint_limit_violation
        self.use_iteration_userdata = getattr(mj_model, "nuserdata", 0) > 0
        self.userdata_index_base_ori = 0 if self.use_iteration_userdata else None
        self.lb = mj_model.actuator_ctrlrange[:, 0].copy()
        self.ub = mj_model.actuator_ctrlrange[:, 1].copy()

        # mj_model.jnt_limited[:] = 0
        # mj_model.jnt_range[:]   = [-jnp.inf, jnp.inf]

        mj_model.actuator_forcelimited[:] = 0
        mj_model.actuator_ctrllimited[:]  = 0
        mj_model.actuator_ctrlrange[:]    = [-jnp.inf, jnp.inf]

        super().__init__(
            mj_model,
            trace_sites=["imu_in_torso", "left_foot", "right_foot"],
        )

        # Get sensor IDs
        self.left_foot_pos_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_position"
        )
        self.left_foot_quat_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_orientation"
        )
        self.right_foot_pos_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_position"
        )
        self.right_foot_quat_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_orientation"
        )

        # Download and load reference data
        npz_file = np.load(
            hf_hub_download(
                repo_id="robfiras/loco-mujoco-datasets",
                filename=reference_filename,
                repo_type="dataset",
            )
        )

        print(f'starting frame:{start_frame}')
        reference = npz_file["qpos"]
        # reference = jnp.array(reference)
        start_config = reference[start_frame]
        self.reference_fps = 40

        # Precompute start foot positions and orientations
        mj_data = mujoco.MjData(mj_model)
        self.start_config = jnp.array(start_config)
        mj_data.qpos[:] = self.start_config

        # initilize array, will be populated later via rotation matrix to
        # quaternion conversion
        ref_left_quat = np.zeros(4)
        ref_right_quat = np.zeros(4)

        mujoco.mj_forward(mj_model, mj_data)
        ref_left_pos = mj_data.site_xpos[mj_model.site("left_foot").id]
        ref_right_pos = mj_data.site_xpos[mj_model.site("right_foot").id]

        # rotation matrix to quaternion
        mujoco.mju_mat2Quat(
            ref_left_quat,
            mj_data.site_xmat[mj_model.site("left_foot").id].flatten(),
        )
        mujoco.mju_mat2Quat(
            ref_right_quat,
            mj_data.site_xmat[mj_model.site("right_foot").id].flatten(),
        )

        self.path_length = 0.4
        foot_heigth = 0.05
        total_time = 2
        n_frame = int(self.reference_fps * total_time)
        self.n_frame = n_frame
        dt = 1 / self.reference_fps
        # import pdb; pdb.set_trace()

        self.ref_left_pos = np.zeros((n_frame, 3))
        self.ref_right_pos = np.zeros((n_frame, 3))
        self.ref_config = np.zeros((n_frame, len(start_config)))
        for i in range(n_frame):
            t = dt * i
            self.ref_config[i] = start_config
            self.ref_config[i][0] += t * self.path_length / 2
            if t < 1:
                self.ref_left_pos[i] = ref_left_pos
                # overwrite x position
                self.ref_left_pos[i][0] += t * self.path_length
                # foot lifting
                self.ref_left_pos[i][2] += foot_heigth * np.sin(t * np.pi)

                self.ref_right_pos[i] = ref_right_pos

            else:  # second half of the gait cycle
                self.ref_left_pos[i] = ref_left_pos
                # left foot planted each each frame $i$,
                # note self.ref_left_pos is 0
                self.ref_left_pos[i][0] += self.path_length

                self.ref_right_pos[i] = ref_right_pos
                # rebase time to start from the same phase in pi space
                self.ref_right_pos[i][0] += (t - 1) * self.path_length
                self.ref_right_pos[i][2] += foot_heigth * np.sin(
                    (t - 1) * np.pi)

        self.goal = reference[start_frame]
        self.goal[0] += self.path_length  # moved forward
        self.goal = jnp.array(self.goal)
        mj_data.qpos[:] = self.goal
        mujoco.mj_forward(mj_model, mj_data)


        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(self.ref_left_pos[:, 0], label="x")
        # plt.plot(self.ref_left_pos[:, 1], label="y")
        # plt.plot(self.ref_left_pos[:, 2], label="z")
        # plt.legend()
        # plt.title("left")

        # plt.figure()
        # plt.plot(self.ref_right_pos[:, 0], label="x")
        # plt.plot(self.ref_right_pos[:, 1], label="y")
        # plt.plot(self.ref_right_pos[:, 2], label="z")
        # plt.legend()
        # plt.title("right")
        # plt.show()



        # Convert reference data to jax arrays
        self.ref_left_pos = jnp.array(self.ref_left_pos)
        self.ref_right_pos = jnp.array(self.ref_right_pos)

        self.ref_right_quat = jnp.array(ref_right_quat)
        self.ref_left_quat = jnp.array(ref_left_quat)

        self.ref_config = jnp.array(self.ref_config)

        # Cost weights
        cost_weights = 1 * np.ones(mj_model.nq)
        cost_weights[:3] = 10.0  # Base pose is more important
        self.cost_weights = jnp.array(cost_weights)
        self.flag_zero_running_cost = flag_zero_running_cost

    def _get_reference_configuration(self, t: jax.Array) -> jax.Array:
        """Get the reference position (q) at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.n_frame - 1)
        return self.ref_config[i]

    def _get_reference_foot_data(self, t: jax.Array) -> tuple[jax.Array, ...]:
        """Get the reference foot positions and orientations at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.n_frame - 1)
        return (
            self.ref_left_pos[i],
            self.ref_left_quat,
            self.ref_right_pos[i],
            self.ref_right_quat,
        )

    def _get_foot_position_errors(
        self, state: mjx.Data
    ) -> tuple[jax.Array, jax.Array]:
        """Get position errors for both feet."""
        ref_left_pos, _, ref_right_pos, _ = self._get_reference_foot_data(
            state.time
        )

        left_pos_adr = self.model.sensor_adr[self.left_foot_pos_sensor]
        right_pos_adr = self.model.sensor_adr[self.right_foot_pos_sensor]

        left_err = (
            state.sensordata[left_pos_adr : left_pos_adr + 3] - ref_left_pos
        )
        right_err = (
            state.sensordata[right_pos_adr : right_pos_adr + 3] - ref_right_pos
        )

        return left_err, right_err

    def _get_foot_orientation_errors(
        self, state: mjx.Data
    ) -> tuple[jax.Array, jax.Array]:
        """Get orientation errors for both feet."""
        _, ref_left_quat, _, ref_right_quat = self._get_reference_foot_data(
            state.time
        )

        left_quat_adr = self.model.sensor_adr[self.left_foot_quat_sensor]
        right_quat_adr = self.model.sensor_adr[self.right_foot_quat_sensor]

        left_quat = state.sensordata[left_quat_adr : left_quat_adr + 4]
        right_quat = state.sensordata[right_quat_adr : right_quat_adr + 4]

        left_err = quat_sub(left_quat, ref_left_quat)
        right_err = quat_sub(right_quat, ref_right_quat)

        return left_err, right_err

    def _get_foot_positions(self, state: mjx.Data) -> tuple[jax.Array, jax.Array]:
        """Get measured foot positions from sensors."""
        left_pos_adr = self.model.sensor_adr[self.left_foot_pos_sensor]
        right_pos_adr = self.model.sensor_adr[self.right_foot_pos_sensor]
        left_pos = state.sensordata[left_pos_adr : left_pos_adr + 3]
        right_pos = state.sensordata[right_pos_adr : right_pos_adr + 3]
        return left_pos, right_pos

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        if self.flag_zero_running_cost:
            return jnp.array(0.0)

        # Configuration error weighs the base pose more heavily
        q_ref = self._get_reference_configuration(state.time)
        q = state.qpos
        full_state_configuration_cost = self._full_state_configuration_cost(q, q_ref)

        # Foot tracking costs
        left_pos_err, right_pos_err = self._get_foot_position_errors(state)
        left_ori_err, right_ori_err = self._get_foot_orientation_errors(state)

        foot_position_cost = jnp.sum(jnp.square(left_pos_err)) + jnp.sum(
            jnp.square(right_pos_err)
        )
        foot_orientation_cost = jnp.sum(jnp.square(left_ori_err)) + jnp.sum(
            jnp.square(right_ori_err)
        )

        # Control penalty
        u_ref = q_ref[7:]
        control_cost = jnp.sum(jnp.square(control - u_ref))

        return (
            10. * full_state_configuration_cost
            + 200. * foot_position_cost
            + 0 * foot_orientation_cost
            + 0.0 * control_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        q_ref = self._get_reference_configuration(state.time)
        q = state.qpos
        full_state_configuration_cost = self._full_state_configuration_cost(q, q_ref)

        (
            base_pos_cost,
            base_ori_cost,
            _ori_weight,
            _base_se3_cost,
            _unweighted_sum,
        ) = self.base_cost_terms(state)

        # Add foot tracking costs to terminal cost
        left_pos_err, right_pos_err = self._get_foot_position_errors(state)
        left_ori_err, right_ori_err = self._get_foot_orientation_errors(state)
        left_pos, right_pos = self._get_foot_positions(state)

        foot_position_cost = jnp.sum(jnp.square(left_pos_err)) + jnp.sum(
            jnp.square(right_pos_err)
        )
        foot_orientation_cost = jnp.sum(jnp.square(left_ori_err)) + jnp.sum(
            jnp.square(right_ori_err)
        )

        base_height_cost = jnp.square(q[2] - q_ref[2])
        support_cost = support_polygon_cost(
            q[:2],
            jnp.stack([left_pos, right_pos]),
            support_margin=self.stand_support_margin,
        )
        fall_penalty = jnp.where(
            is_standing_stable(
                q,
                state.qvel,
                q_ref,
                base_pos_cost_tol=self.stand_base_pos_cost_tol,
                base_ori_cost_tol=self.stand_base_ori_cost_tol,
                min_base_height=self.stand_min_base_height,
                max_base_lin_vel=self.stand_max_base_lin_vel,
                max_base_ang_vel=self.stand_max_base_ang_vel,
                foot_positions=jnp.stack([left_pos, right_pos]), # sensor
                support_margin=self.stand_support_margin,
                joint_range=self.model.jnt_range
                if self.stand_joint_limit_violation is not None
                else None,
                max_joint_limit_violation=self.stand_joint_limit_violation,
            ),
            0.0,
            self.terminal_fall_penalty,
        )
        weighted_cost = self._terminal_cost_terms(
            full_state_configuration_cost,
            base_pos_cost,
            base_ori_cost,
            foot_position_cost,
            foot_orientation_cost,
            base_height_cost,
            support_cost,
        )
        return weighted_cost + fall_penalty

    def terminal_cost_unweighted(self, state: mjx.Data) -> jax.Array:
        """Unweighted terminal cost (sum of raw cost terms)."""
        q_ref = self._get_reference_configuration(state.time)
        q = state.qpos
        full_state_configuration_cost = jnp.sum(jnp.square(q - q_ref))

        (
            base_pos_cost,
            base_ori_cost,
            _ori_weight,
            _base_se3_cost,
            _unweighted_sum,
        ) = self.base_cost_terms(state)

        left_pos_err, right_pos_err = self._get_foot_position_errors(state)
        left_ori_err, right_ori_err = self._get_foot_orientation_errors(state)
        left_pos, right_pos = self._get_foot_positions(state)

        foot_position_cost = jnp.sum(jnp.square(left_pos_err)) + jnp.sum(
            jnp.square(right_pos_err)
        )
        foot_orientation_cost = jnp.sum(jnp.square(left_ori_err)) + jnp.sum(
            jnp.square(right_ori_err)
        )
        base_height_cost = jnp.square(q[2] - q_ref[2])
        support_cost = support_polygon_cost(
            q[:2],
            jnp.stack([left_pos, right_pos]),
            support_margin=self.stand_support_margin,
        )
        return (
            full_state_configuration_cost
            + base_pos_cost
            + base_ori_cost
            + foot_position_cost
            + foot_orientation_cost
            + base_height_cost
            + support_cost
        )

    def _terminal_cost_terms(
        self,
        full_state_configuration_cost: jax.Array,
        base_pos_cost: jax.Array,
        base_ori_cost: jax.Array,
        foot_position_cost: jax.Array,
        foot_orientation_cost: jax.Array,
        base_height_cost: jax.Array,
        support_cost: jax.Array,
    ) -> jax.Array:
        if self.terminal_cost_controller is None:
            return (
                self.terminal_full_config_cost_weight * full_state_configuration_cost
                + self.terminal_base_se3_weight * (base_pos_cost + self.base_ori_weight * base_ori_cost)
                + self.terminal_foot_position_weight * foot_position_cost
                + self.terminal_foot_position_weight * foot_orientation_cost
            )
        weights = self.terminal_cost_controller.weights
        return (
            self.terminal_full_config_cost_weight * full_state_configuration_cost
            + weights.base_pos_weight * base_pos_cost
            + weights.base_ori_weight * base_ori_cost
            + weights.foot_pos_weight * foot_position_cost
            + weights.foot_ori_weight * foot_orientation_cost
            + weights.base_height_weight * base_height_cost
            + weights.support_weight * support_cost
        )

    def _full_state_configuration_cost(self, q: jax.Array, q_ref: jax.Array) -> jax.Array:
        """Weighted full-state configuration cost."""
        q_err = self.cost_weights * (q - q_ref)
        return jnp.sum(jnp.square(q_err))

    def base_cost_terms(
        self, state: mjx.Data
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return base position/orientation costs and combined terms."""
        q_ref = self._get_reference_configuration(state.time)
        q = state.qpos
        if self.use_se3_twist and self._se3_twist_calc is not None:
            twist = self._se3_twist_calc.from_qpos(q, q_ref)
            base_pos_cost = jnp.sum(jnp.square(twist[:3]))
            base_ori_cost = jnp.sum(jnp.square(twist[3:]))
        else:
            base_pos_err = q[:3] - q_ref[:3]
            base_ori_err = quat_sub(q[3:7], q_ref[3:7])
            base_pos_cost = jnp.sum(jnp.square(base_pos_err))
            base_ori_cost = jnp.sum(jnp.square(base_ori_err))
        if self.use_iteration_userdata:
            ori_weight = state.userdata[self.userdata_index_base_ori]
        else:
            ori_weight = self.base_ori_weight
        weighted_sum = base_pos_cost + ori_weight * base_ori_cost
        unweighted_sum = base_pos_cost + base_ori_cost
        return base_pos_cost, base_ori_cost, ori_weight, weighted_sum, unweighted_sum

    def set_iteration_progress(self, iteration: int, max_iteration: int) -> jax.Array:
        """Linearly ramp orientation weight from start to end over iterations."""
        if max_iteration <= 1:
            progress = 1.0
        else:
            progress = float(iteration) / float(max_iteration - 1)
        progress = min(max(progress, 0.0), 1.0)
        weight = self.base_ori_weight_start + progress * (
            self.base_ori_weight_end - self.base_ori_weight_start
        )
        if self.base_weight_scheduler is not None:
            weight = self.base_weight_scheduler.update(iteration, float(weight))
        self.base_ori_weight = jnp.asarray(weight)
        if self.terminal_cost_controller is not None:
            self.terminal_cost_controller.update(iteration)
        return self.base_ori_weight

    def set_base_cost_log_path(self, path: str | None) -> None:
        if self.base_weight_scheduler is not None:
            self.base_weight_scheduler.set_cost_log_path(path)

    def set_terminal_cost_log_path(self, path: str | None) -> None:
        if self.terminal_cost_controller is not None:
            self.terminal_cost_controller.set_cost_log_path(path)

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the friction parameters."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,),
                                        minval=0.5, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly perturb the measured base position and velocities."""
        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_err = 0.01 * jax.random.normal(q_rng, (7,))
        v_err = 0.01 * jax.random.normal(v_rng, (6,))

        qpos = data.qpos.at[0:7].set(data.qpos[0:7] + q_err)
        qvel = data.qvel.at[0:6].set(data.qvel[0:6] + v_err)

        return {"qpos": qpos, "qvel": qvel}
