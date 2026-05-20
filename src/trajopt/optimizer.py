import time
import csv
import os
import threading
import subprocess
import tracemalloc
import jax
jax.config.update("jax_compilation_cache_dir", "tmp/jax_cache") # hack to avoid requiring sudo access to /tmp folder
import jax.numpy as jnp
import utils.mujoco_env as mujoco_env  # sets MUJOCO_GL early for headless runs
# NOTE: Don't set MUJOCO_GL here. MuJoCo selects/loads its OpenGL backend at the
# first `import mujoco` in the process, and `runners.compare` (and other modules) may
# import `mujoco` before `traj_opt.py` is imported; setting it here would then be
# too late. If you want to force offscreen rendering, set it in your shell:
# `MUJOCO_GL=egl ...` or `MUJOCO_GL=osmesa ...`.
# os.environ["MUJOCO_GL"] = "egl"
import mujoco

import numpy as np
from mujoco import mjx
from hydrax.alg_base import Trajectory, SamplingBasedController
import tqdm
from functools import partial
from pathlib import Path
from datetime import datetime
import traceback
import gc
from utils.log_utils import log_line
from algs.cbo_stats import CBOxStatsRecorder, update_best_ever_cost
from trajopt.artifacts import RunArtifacts
from trajopt.gif_utils import save_consensus_gif, save_ranked_particle_gifs
from trajopt.helpers import save_mj_data_state
from trajopt.visuals import TrajectoryOptimizerVisuals
from utils.params_io import ParamsIO


class TrajectoryOptimizer:

    def __init__(
        self,
        name: str,
        controller: SamplingBasedController,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        max_eval: int | None = None,
        use_jit: bool = True,
        config_path: Path | None = None,
    ):


        self.warm_up = False
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.controller = controller
        self.use_jit = use_jit


        mjx_data = mjx.put_data(self.mj_model, self.mj_data)
        mjx_data = mjx_data.replace(mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat)
        self.mjx_data = mjx_data

        self.viewer = None
        self.visuals = TrajectoryOptimizerVisuals(self)
        self.controller_name = name

        # Limit the number of rollouts evaluated at once to avoid large GPU allocations
        self.max_eval = max_eval
        self.run_dir: Path | None = None
        self.artifacts = RunArtifacts(
            controller_name=name,
            controller=controller,
            config_path=config_path,
        )
        self.memory_log_every = int(os.getenv("MEMORY_LOG_EVERY", "50"))
        self.mean_knots_cost_every = int(os.getenv("MEAN_KNOTS_COST_EVERY", "1"))
        self.loss_log_every = int(os.getenv("MEAN_KNOTS_COST_EVERY", "0"))
        self.gc_every = int(os.getenv("GC_EVERY", "0"))
        self.jax_clear_every = int(os.getenv("JAX_CLEAR_EVERY", "0"))
        self.tracemalloc_every = int(os.getenv("TRACEMALLOC_EVERY", "0"))
        self.config_path = config_path

        if controller is not None:
            self.params_io = ParamsIO(controller)

            # initialize the controller
            if use_jit:
                jit_optimize = jax.jit(partial(controller.optimize))
                self.jit_optimize = jit_optimize
            else:
                self.jit_optimize = controller.optimize

            # logging
            print(
                f"Trajectory Optimization with {controller.num_knots} steps "
                f"over a {controller.ctrl_steps * controller.task.dt} "
                f"second horizon."
            )

            print(f'task.dt:{controller.task.dt}; controller.dt:{controller.dt}; '
                f'task.model.opt.timestep: {controller.task.model.opt.timestep}; '
                f'task.mj_model.opt.timestep: {controller.task.mj_model.opt.timestep}; '
                f'simulator mj_model.opt.timestep: {mj_model.opt.timestep}'
                )

    def _maybe_start_gpu_monitor(self):
        class _GpuMonitor:
            def __init__(self, interval_s: float = 0.5, gpu_index: int = 0) -> None:
                self.interval_s = interval_s
                self.gpu_index = gpu_index
                self._running = False
                self._thread = None
                self.max_used_mb = 0

            def _sample_once(self) -> None:
                try:
                    out = subprocess.check_output(
                        [
                            "nvidia-smi",
                            f"--query-gpu=memory.used",
                            "--format=csv,noheader,nounits",
                            f"--id={self.gpu_index}",
                        ],
                        text=True,
                    ).strip()
                    if out:
                        used_mb = int(out.splitlines()[0])
                        if used_mb > self.max_used_mb:
                            self.max_used_mb = used_mb
                except Exception:
                    return

            def _run(self) -> None:
                while self._running:
                    self._sample_once()
                    time.sleep(self.interval_s)

            def start(self) -> None:
                self._running = True
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

            def stop(self) -> int:
                self._running = False
                if self._thread is not None:
                    self._thread.join(timeout=2.0)
                return self.max_used_mb

        interval_s = float(os.getenv("GPU_MONITOR_INTERVAL_S", "0.5"))
        gpu_index = int(os.getenv("GPU_MONITOR_INDEX", "0"))
        monitor = _GpuMonitor(interval_s=interval_s, gpu_index=gpu_index)
        monitor.start()
        return monitor

    def _read_terminal_base_height(self, iteration: int) -> float | None:
        if self.run_dir is None:
            return None
        term_path = self.run_dir / "stats" / "terminal_costs.csv"
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

    def _save_particles_gif(
        self,
        mean_knots: jax.Array,
        iteration: int,
        particles: jax.Array,
        rollout_costs: np.ndarray,
        *,
        persist_latest: bool,
        persist_every: bool,
    ) -> None:
        self.visuals.save_particles_gif(
            mean_knots=mean_knots,
            iteration=iteration,
            particles=particles,
            rollout_costs=rollout_costs,
            persist_latest=persist_latest,
            persist_every=persist_every,
        )

    def _persist_consensus_knots(
        self,
        mean_knots: jax.Array,
        iteration: int,
        *,
        persist_latest: bool,
        persist_every: bool,
    ) -> None:
        particles_dir = None
        if hasattr(self.controller, "io"):
            particles_dir = getattr(self.controller.io, "_particles_dir", None)
        if particles_dir is None:
            particles_dir = getattr(self.controller, "_particles_dir", None)
        if particles_dir is None:
            return
        try:
            consensus_np = np.asarray(jax.device_get(mean_knots))
        except Exception:
            consensus_np = np.asarray(mean_knots)
        if persist_latest:
            np.save(Path(particles_dir) / "consensus_latest.npy", consensus_np)
            (Path(particles_dir) / "consensus_latest_iter.txt").write_text(
                f"{iteration}\n", encoding="utf-8"
            )
        if persist_every:
            np.save(
                Path(particles_dir) / f"consensus_iter_{iteration:05d}.npy",
                consensus_np,
            )

    def _persist_mj_data(
        self,
        *,
        iteration: int,
        persist_latest: bool,
        persist_every: bool,
    ) -> None:
        particles_dir = None
        if hasattr(self.controller, "io"):
            particles_dir = getattr(self.controller.io, "_particles_dir", None)
        if particles_dir is None:
            particles_dir = getattr(self.controller, "_particles_dir", None)
        if particles_dir is None:
            return
        particles_dir = Path(particles_dir)
        particles_dir.mkdir(parents=True, exist_ok=True)
        if persist_latest:
            save_mj_data_state(particles_dir / "mjdata_latest.npz", self.mj_data)
            (particles_dir / "mjdata_latest_iter.txt").write_text(
                f"{iteration}\n", encoding="utf-8"
            )
        if persist_every:
            save_mj_data_state(
                particles_dir / f"mjdata_iter_{iteration:05d}.npz",
                self.mj_data,
            )

    def reset_mjx_data(self):
        """Enhanced reset method that ensures complete state reset"""
        # Create a fresh mjx_data from the original mj_data
        mjx_data = mjx.put_data(self.mj_model, self.mj_data)
        mjx_data = mjx_data.replace(
            mocap_pos=self.mj_data.mocap_pos,
            mocap_quat=self.mj_data.mocap_quat,
            time=self.mj_data.time,  # Ensure time is also reset
            qpos = self.mj_data.qpos,
            qvel = self.mj_data.qvel
        )
        self.mjx_data = mjx_data

    def _create_run_dir(self) -> Path:
        base_dir = Path("tmp/experiments")
        base_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = base_dir / timestamp
        suffix = 0
        while candidate.exists():
            suffix += 1
            candidate = base_dir / f"{timestamp}_{suffix:02d}"
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def _get_optimize_runtime_log_path(self) -> Path | None:
        if self.run_dir is None:
            return None
        return self.run_dir / "stats" / "optimize_runtime.csv"

    def _log_optimize_runtime(
        self,
        *,
        iteration: int,
        elapsed_s: float,
        phase: str = "execute",
    ) -> None:
        log_path = self._get_optimize_runtime_log_path()
        if log_path is None:
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists()
        with log_path.open("a", encoding="utf-8") as f:
            if write_header:
                f.write("iteration,phase,elapsed_s,elapsed_ms\n")
            f.write(f"{iteration},{phase},{elapsed_s:.9f},{elapsed_s * 1000.0:.6f}\n")

    def optimize(
        self,
        max_iteration: int = 100,
        seed: int = 1,
        initial_knots: jax.Array = None
    ) -> list[list, list, Trajectory]:

        cost_list = []

        self.reset_mjx_data()
        monitor = self._maybe_start_gpu_monitor()
        try:
            if self.run_dir is None:
                self.run_dir = self._create_run_dir()
            else:
                self.run_dir.mkdir(parents=True, exist_ok=True)
            self.artifacts.set_run_dir(self.run_dir)
            if hasattr(self, "params_io"):
                self.params_io._params_dir = self.run_dir / "params"
            self.artifacts.mark_status_started()
            self.artifacts.dump_run_metadata()
            if self.controller is not None:
                setattr(self.controller, "experiment_start_time", datetime.now())
                if hasattr(self.controller, "io"):
                    setattr(self.controller.io, "_particles_dir", self.run_dir / "particles")
                else:
                    setattr(self.controller, "_particles_dir", self.run_dir / "particles")
                setattr(self.controller, "stats_log_path", self.artifacts.stats_log_path)
                setattr(self.controller, "loss_log_path", self.artifacts.loss_log_path)
                setattr(self.controller, "cost_log_path", self.artifacts.cost_log_path)
                setattr(self.controller, "base_cost_log_path", self.run_dir / "stats" / "base_costs.csv")
                setattr(self.controller, "terminal_cost_log_path", self.run_dir / "stats" / "terminal_costs.csv")
                setattr(self.controller, "last_output_path", self.artifacts.get_last_output_path())
                if hasattr(self.controller.task, "set_base_cost_log_path"):
                    self.controller.task.set_base_cost_log_path(
                        str(self.run_dir / "stats" / "base_costs.csv")
                    )
                if hasattr(self.controller.task, "set_terminal_cost_log_path"):
                    self.controller.task.set_terminal_cost_log_path(
                        str(self.run_dir / "stats" / "terminal_costs.csv")
                    )
                setattr(self.controller, "_init_state_mjx_data", self.mjx_data)

            policy_params = self.controller.init_params(
                seed=seed, initial_knots=initial_knots)
            mean_knots = policy_params.mean

            if self.use_jit and hasattr(self.jit_optimize, "lower"):
                compile_t0 = time.perf_counter()
                self.jit_optimize = self.jit_optimize.lower(
                    self.mjx_data, policy_params
                ).compile()
                compile_elapsed_s = time.perf_counter() - compile_t0
                self._log_optimize_runtime(
                    iteration=-1,
                    elapsed_s=compile_elapsed_s,
                    phase="compile",
                )
                log_line(
                    f"jit compile time for optimize: {compile_elapsed_s * 1000.0:.3f} ms",
                    self.artifacts.get_last_output_path(),
                )

            start_iteration = 0
            loaded_iter = getattr(self.controller, "loaded_particles_iteration", None)
            if loaded_iter is not None:
                start_iteration = loaded_iter + 1
            stats_recorder = CBOxStatsRecorder(self.controller)
            for i_generation in tqdm.tqdm(range(start_iteration, start_iteration + max_iteration)):
                if self.memory_log_every > 0 and i_generation % self.memory_log_every == 0:
                    self.artifacts.log_memory_usage(i_generation)
                if self.gc_every > 0 and i_generation % self.gc_every == 0:
                    gc.collect()
                if self.jax_clear_every > 0 and i_generation % self.jax_clear_every == 0:
                    try:
                        jax.clear_caches()
                    except Exception:
                        pass
                if self.tracemalloc_every > 0 and i_generation % self.tracemalloc_every == 0:
                    if not tracemalloc.is_tracing():
                        tracemalloc.start()
                    self.artifacts.log_tracemalloc(i_generation)
                if hasattr(self.controller.task, "set_iteration_progress"):
                    base_ori_weight = self.controller.task.set_iteration_progress(
                        i_generation, max_iteration
                    )
                    if getattr(self.controller.task, "use_iteration_userdata", False):
                        if hasattr(self.mjx_data, "userdata") and self.mjx_data.userdata.size > 0:
                            self.mjx_data = self.mjx_data.replace(
                                userdata=self.mjx_data.userdata.at[0].set(base_ori_weight)
                            )
                if self.controller_name == "CBOxPolarized" or getattr(self.controller, "controller_name", None) == "CBOxPolarized":
                    setattr(self.controller, "_iteration_for_stats", i_generation)
                t_optimize_start = time.perf_counter()
                policy_params, rollouts = self.jit_optimize(
                    self.mjx_data, policy_params
                )
                policy_params, rollouts = jax.block_until_ready((policy_params, rollouts))
                optimize_elapsed_s = time.perf_counter() - t_optimize_start
                self._log_optimize_runtime(
                    iteration=i_generation,
                    elapsed_s=optimize_elapsed_s,
                )

                rollout_costs = np.asarray(jnp.sum(rollouts.costs, axis=1))
                best_cost = float(rollout_costs.min())
                mean_cost = float(rollout_costs.mean())
                best_ever_cost, best_ever_iter = update_best_ever_cost(
                    self.controller, best_cost, i_generation
                )
                cost_list.append(best_cost)
                if hasattr(self.controller, "stats") and hasattr(self.controller.stats, "log_iteration"):
                    self.controller.stats.log_iteration(
                        i_generation,
                        policy_params,
                        rollouts,
                        best_cost,
                        mean_cost,
                        state_mjx_data=self.mjx_data,
                    )
                elif hasattr(self.controller, "log_iteration"):
                    self.controller.log_iteration(
                        i_generation,
                        policy_params,
                        rollouts,
                        best_cost,
                        mean_cost,
                        state_mjx_data=self.mjx_data,
                    )
                else:
                    log_line(
                        f"best/mean/best_ever cost: {best_cost:.6f}/{mean_cost:.6f}/{best_ever_cost:.6f}; "
                        f"best_ever iter: {best_ever_iter}",
                        self.artifacts.get_last_output_path(),
                    )
                if (
                    self.loss_log_every > 0
                    and i_generation % self.loss_log_every == 0
                ):
                    stats_recorder._persist_loss_stats(
                        i_generation,
                        best_cost,
                        mean_cost,
                    )
                    stats_recorder._persist_cost_stats(
                        i_generation,
                        best_cost,
                        mean_cost,
                        best_ever_cost,
                    )
                particles_to_persist = None
                if hasattr(rollouts, "knots"):
                    particles_to_persist = rollouts.knots
                elif hasattr(policy_params, "particles"):
                    particles_to_persist = policy_params.particles
                if particles_to_persist is not None and hasattr(self.controller, "io"):
                    persistence_temperature = getattr(policy_params, "temperature", None)
                    persist_latest, persist_every = self.controller.io.persist_particles(
                        particles_to_persist,
                        rollout_costs,
                        iteration=i_generation + 1,
                        temperature=persistence_temperature,
                    )
                    if persist_latest or persist_every:
                        self._persist_consensus_knots(
                            policy_params.mean,
                            iteration=i_generation + 1,
                            persist_latest=persist_latest,
                            persist_every=persist_every,
                        )
                        self._persist_mj_data(
                            iteration=i_generation + 1,
                            persist_latest=persist_latest,
                            persist_every=persist_every,
                        )
                        self._save_particles_gif(
                            policy_params.mean,
                            iteration=i_generation + 1,
                            particles=particles_to_persist,
                            rollout_costs=rollout_costs,
                            persist_latest=persist_latest,
                            persist_every=persist_every,
                        )
                if hasattr(self, "params_io"):
                    self.params_io.persist_params(policy_params, iteration=i_generation + 1)
                self.reset_mjx_data()
                mean_knots = policy_params.mean
                if self.mean_knots_cost_every > 0 and i_generation % self.mean_knots_cost_every == 0:
                    # Evaluate only the latest mean to keep memory use low during logging
                    cost_eval, _ = self.get_cost_control_list([mean_knots])
                    stats_recorder._persist_cost_stats(
                        i_generation,
                        best_cost,
                        mean_cost,
                        best_ever_cost,
                    )
                    log_line(
                        f"iteration {i_generation} cost: {cost_eval[-1]:.6f}",
                        self.artifacts.get_last_output_path(),
                    )

            _, controls = self.get_cost_control_list([mean_knots])
            self.artifacts.mark_status_completed()
            return cost_list, controls
        except Exception as exc:
            self.artifacts.mark_status_failed(exc)
            raise
        finally:
            if monitor is not None:
                max_used_mb = monitor.stop()
                log_line(
                    f"GPU max memory used during optimize: {max_used_mb} MiB",
                    self.artifacts.get_last_output_path(),
                )
                if self.run_dir is not None:
                    gpu_log_path = self.run_dir / "stats" / "gpu_max_memory.txt"
                    gpu_log_path.parent.mkdir(parents=True, exist_ok=True)
                    gpu_log_path.write_text(f"{max_used_mb}\n", encoding="utf-8")

    def get_cost_control_list(
        self,
        knots_list: list,
        batch_size: int | None = None,
    ) -> list:

        ctrl = self.controller
        task = self.controller.task

        batch_size = batch_size or self.max_eval or len(knots_list)
        batch_size = max(batch_size, 1)

        knots = jnp.array(knots_list)

        cost_batches = []
        controls_batches = []

        # Process rollouts in smaller batches to avoid large allocations on the GPU
        for start in range(0, knots.shape[0], batch_size):
            end = min(start + batch_size, knots.shape[0])
            knots_batch = knots[start:end]

            controls_batch = self.knots2ctrls(knots_batch)

            state = self.mjx_data
            _, rollouts = ctrl.eval_rollouts(task.model, state, controls_batch, knots_batch)

            cost_batches.append(np.array(jnp.sum(rollouts.costs, axis=-1)))
            controls_batches.append(np.array(controls_batch))

        costs = np.concatenate(cost_batches, axis=0)
        controls = np.concatenate(controls_batches, axis=0)

        return list(costs), controls

    def knots2ctrls(self,
                    knots: jax.Array
        )-> jax.Array:
        # This function follow exactly how spline interpolation was done in Hydrax: https://github.com/vincekurtz/hydrax/blob/main/hydrax/alg_base.py/#L208

        ctrl = self.controller

        tk = (
            jnp.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots) + self.mjx_data.time
        )

        tq = jnp.linspace(tk[0], tk[-1], ctrl.ctrl_steps)
        # Hydrax's interpolation function is vmapped over the rollout axis, so
        # knots should be shaped (num_rollouts, num_knots, nu). For convenience,
        # accept both a single trajectory (num_knots, nu) and common transpose
        # variants when loading from disk.
        knots = jnp.asarray(knots)

        if knots.ndim == 2:
            if knots.shape == (ctrl.num_knots, self.mj_model.nu):
                knots_batch = knots[None, ...]
            elif knots.shape == (self.mj_model.nu, ctrl.num_knots):
                knots_batch = knots.T[None, ...]
            else:
                raise ValueError(
                    "Expected knots shape "
                    f"({ctrl.num_knots},{self.mj_model.nu}) or "
                    f"({self.mj_model.nu},{ctrl.num_knots}), got {tuple(knots.shape)}"
                )
            controls = ctrl.interp_func(tq, tk, knots_batch)[0]
        elif knots.ndim == 3:
            if knots.shape[1:] == (ctrl.num_knots, self.mj_model.nu):
                knots_batch = knots
            elif knots.shape[1:] == (self.mj_model.nu, ctrl.num_knots):
                knots_batch = jnp.swapaxes(knots, 1, 2)
            else:
                raise ValueError(
                    "Expected knots shape "
                    f"(B,{ctrl.num_knots},{self.mj_model.nu}) or "
                    f"(B,{self.mj_model.nu},{ctrl.num_knots}), got {tuple(knots.shape)}"
                )
            controls = ctrl.interp_func(tq, tk, knots_batch)
        else:
            raise ValueError(f"Expected knots with 2 or 3 dims, got {tuple(knots.shape)}")

        return controls
