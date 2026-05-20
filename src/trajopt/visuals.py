from __future__ import annotations

import copy
import time
from pathlib import Path

import jax
import mujoco
import numpy as np
import imageio
import glfw
import matplotlib.pyplot as plt

from utils.log_utils import log_line
from trajopt.gif_utils import save_consensus_gif, save_ranked_particle_gifs
from trajopt.helpers import (
    compute_gif_playback_settings,
    build_stride_frame_indices,
    apply_mj_data_state,
    load_mj_data_state,
)


class TrajectoryOptimizerVisuals:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer

    def save_particles_gif(
        self,
        mean_knots: jax.Array,
        iteration: int,
        particles: jax.Array,
        rollout_costs: np.ndarray,
        *,
        persist_latest: bool,
        persist_every: bool,
    ) -> None:
        optimizer = self.optimizer
        if optimizer.run_dir is None:
            return
        if persist_latest or persist_every:
            log_line(
                f"persisting mjdata for gifs at iteration {iteration}",
                optimizer.artifacts.get_last_output_path(),
            )
            optimizer._persist_mj_data(
                iteration=iteration,
                persist_latest=persist_latest,
                persist_every=persist_every,
            )
        effective_playback_dt, frame_stride, playback_suffix, _ = compute_gif_playback_settings(
            optimizer.mj_model.opt.timestep,
            playback_speed=8.0,
            frame_stride=1,
        )
        save_consensus_gif(
            optimizer,
            mean_knots=mean_knots,
            iteration=iteration,
            playback_suffix=playback_suffix,
            playback_dt=effective_playback_dt,
            frame_stride=frame_stride,
        )
        try:
            particles_np = np.asarray(jax.device_get(particles))
        except Exception:
            particles_np = np.asarray(particles)
        if particles_np.ndim != 3 or particles_np.shape[0] == 0:
            return
        save_ranked_particle_gifs(
            optimizer,
            particles_np=particles_np,
            rollout_costs=rollout_costs,
            iteration=iteration,
            playback_suffix=playback_suffix,
            playback_dt=effective_playback_dt,
            frame_stride=frame_stride,
        )

    def normalize_controls_for_mujoco(self, controls: np.ndarray) -> np.ndarray:
        optimizer = self.optimizer
        controls = np.asarray(controls)
        nu = int(optimizer.mj_model.nu)

        if controls.ndim == 1:
            if nu != 1:
                raise ValueError(f"Expected controls with nu={nu}, got 1D {controls.shape}")
            return controls.reshape(-1, 1)

        if controls.ndim != 2:
            raise ValueError(f"Expected controls with 2 dims, got {controls.shape}")

        if controls.shape[1] == nu:
            return controls
        if controls.shape[0] == nu and controls.shape[1] != nu:
            return controls.T

        raise ValueError(f"Expected controls shaped (H,{nu}) or ({nu},H), got {controls.shape}")

    def plot_solution(self, cost_list, controls):
        optimizer = self.optimizer
        ctrls = np.asarray(controls)
        horizon, nu = ctrls.shape

        dt = float(optimizer.controller.dt)
        t = np.arange(horizon) * dt

        ub = getattr(optimizer.controller.task, "ub", None)
        lb = getattr(optimizer.controller.task, "lb", None)

        fig, axes = plt.subplots(nu, 1, sharex=True, figsize=(8, 2 * nu))
        if nu == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(t, ctrls[:, i], lw=1.5)

            if ub is not None and lb is not None:
                ax.axhline(y=ub[i], color="r", linestyle="--", label="ub" if i == 0 else None)
                ax.axhline(y=lb[i], color="r", linestyle="--", label="lb" if i == 0 else None)

            ax.set_ylabel(f"$u_{i + 1}$")
            ax.grid(True)

        task_name = optimizer.controller.task.__class__.__name__

        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        plt.show(block=False)
        plt.savefig("Figures/" + task_name + "_controls.png")

        fig = plt.figure()
        plt.plot(np.array(cost_list))
        plt.grid(True)
        plt.show(block=False)
        plt.savefig("Figures/" + task_name + "_costs.png")

    def visualize_solution(self, controls):
        optimizer = self.optimizer
        self._create_temporary_viewer()

        controls = self.normalize_controls_for_mujoco(controls)
        horizon = controls.shape[0]
        dt = float(optimizer.mj_model.opt.timestep)

        i = 0
        while optimizer.viewer is not None and optimizer.viewer.is_running():
            t_start = time.time()

            optimizer.tmp_mj_data.ctrl[:] = controls[i]
            mujoco.mj_step(optimizer.mj_model, optimizer.tmp_mj_data)
            optimizer.viewer.sync()

            elapsed = time.time() - t_start
            to_sleep = dt - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

            i += 1
            if i == horizon:
                i = 0
                self._reset_tmp_data()

    def _create_temporary_viewer(self):
        optimizer = self.optimizer
        if optimizer.viewer is None:
            try:
                if not glfw.init():
                    print("Warning: GLFW init failed; viewer disabled (check display/driver setup).")
                    return
            except Exception as exc:
                print(f"Warning: GLFW init error ({exc}); viewer disabled.")
                return
            optimizer.tmp_mj_data = copy.copy(optimizer.mj_data)
            optimizer.viewer = mujoco.viewer.launch_passive(
                optimizer.mj_model, optimizer.tmp_mj_data
            )

    def _create_recorder(self) -> bool:
        optimizer = self.optimizer
        optimizer.frames = []
        try:
            optimizer.renderer = mujoco.Renderer(optimizer.mj_model, height=480, width=640)
        except Exception as exc:
            optimizer.renderer = None
            optimizer._render_disabled_reason = str(exc)
            return False
        return True

    def _reset_tmp_data(self):
        optimizer = self.optimizer
        optimizer.tmp_mj_data.qpos[:] = optimizer.mj_data.qpos
        optimizer.tmp_mj_data.qvel[:] = optimizer.mj_data.qvel

    def savegif(
        self,
        controls,
        show_reference: bool = True,
        extra_fname: str = "best_algo",
        out_path: str | Path | None = None,
        playback_dt: float | None = None,
        frame_stride: int = 1,
        frame_indices: list[int] | None = None,
        mjdata_path: str | Path | None = None,
    ):
        optimizer = self.optimizer
        optimizer.tmp_mj_data = copy.copy(optimizer.mj_data)
        if mjdata_path is not None:
            apply_mj_data_state(optimizer.tmp_mj_data, load_mj_data_state(mjdata_path))
            mujoco.mj_forward(optimizer.mj_model, optimizer.tmp_mj_data)
        controls = self.normalize_controls_for_mujoco(controls)
        if not self._create_recorder():
            reason = getattr(optimizer, "_render_disabled_reason", "unknown error")
            print(
                "Warning: skipping gif rendering (MuJoCo OpenGL context not available). "
                f"Set MUJOCO_GL=egl or MUJOCO_GL=osmesa. Underlying error: {reason}"
            )
            return None
        ghost_img = None
        ghost_renderer = None
        try:
            if show_reference and hasattr(optimizer.controller.task, "goal"):
                reference = optimizer.controller.task.goal
                ref_data = mujoco.MjData(optimizer.mj_model)
                ref_data.qpos[:] = reference
                mujoco.mj_forward(optimizer.mj_model, ref_data)
                try:
                    ghost_renderer = mujoco.Renderer(optimizer.mj_model, height=480, width=640)
                    ghost_renderer.update_scene(ref_data)
                    ghost_img = ghost_renderer.render().copy()
                except Exception:
                    ghost_renderer = None
                    ghost_img = None
            horizon = controls.shape[0]
            dt = float(optimizer.mj_model.opt.timestep)
            stride = max(1, int(frame_stride))
            if frame_indices is None:
                frame_indices = build_stride_frame_indices(horizon, stride)
            frame_indices_set = set(frame_indices)
            for i in range(horizon):
                optimizer.tmp_mj_data.ctrl[:] = controls[i]
                mujoco.mj_step(optimizer.mj_model, optimizer.tmp_mj_data)
                if i not in frame_indices_set:
                    continue
                optimizer.renderer.update_scene(optimizer.tmp_mj_data)
                frame = optimizer.renderer.render().copy()
                if show_reference and ghost_img is not None:
                    alpha = 0.3
                    blended_img = (1 - alpha) * frame.astype(np.float32) + alpha * ghost_img.astype(np.float32)
                    img = np.clip(blended_img, 0, 255).astype(np.uint8)
                else:
                    img = frame.copy()
                optimizer.frames.append(img)
            if out_path is None:
                task_name = optimizer.controller.task.__class__.__name__
                out_path = Path("Figures") / f"{task_name}_{extra_fname}_simulation.gif"
            else:
                out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            gif_dt = float(playback_dt) if playback_dt is not None else dt
            gif_dt = max(gif_dt, 0.01)
            imageio.mimsave(out_path.as_posix(), optimizer.frames, duration=gif_dt)
            return out_path
        finally:
            if ghost_renderer is not None:
                try:
                    ghost_renderer.close()
                except Exception:
                    pass
            if getattr(optimizer, "renderer", None) is not None:
                try:
                    optimizer.renderer.close()
                except Exception:
                    pass
                optimizer.renderer = None
