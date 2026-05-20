from __future__ import annotations

from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from utils.humanoid_standing import is_standing_stable, support_polygon_cost
from utils.log_utils import log_line

_LAST_COST_LOG_ITER_BY_PATH: dict[str, int] = {}


def update_best_ever_cost(controller, best_cost: float, iteration: int) -> tuple[float, int]:
    prev_best = getattr(controller, "_best_ever_cost", None)
    prev_best_iter = getattr(controller, "_best_ever_iter", None)
    if prev_best is None or best_cost < prev_best:
        prev_best = best_cost
        prev_best_iter = iteration
    controller._best_ever_cost = prev_best
    if prev_best_iter is None:
        prev_best_iter = iteration
    controller._best_ever_iter = prev_best_iter
    return prev_best, prev_best_iter


class CBOxStatsRecorder:
    def __init__(self, controller) -> None:
        self.controller = controller

    def _get_stats_log_path(self) -> Path:
        return getattr(
            self.controller,
            "stats_log_path",
            Path("tmp/experiments") / "stats" / "consensus_stats.csv",
        )

    def _get_loss_log_path(self) -> Path:
        return getattr(
            self.controller,
            "loss_log_path",
            Path("tmp/experiments") / "stats" / "loss_curve.csv",
        )

    def _get_cost_log_path(self) -> Path | None:
        return getattr(self.controller, "cost_log_path", None)

    def _get_base_cost_log_path(self) -> Path:
        return getattr(
            self.controller,
            "base_cost_log_path",
            Path("tmp/experiments") / "stats" / "base_costs.csv",
        )

    def _get_terminal_cost_log_path(self) -> Path:
        return getattr(
            self.controller,
            "terminal_cost_log_path",
            Path("tmp/experiments") / "stats" / "terminal_costs.csv",
        )

    def _get_last_output_path(self) -> Path:
        return getattr(
            self.controller,
            "last_output_path",
            Path("tmp/experiments") / "stats" / "last_output.txt",
        )

    def _persist_consensus_stats(
        self,
        iteration: int,
        temperature: float,
        consensus_to_best: float,
        consensus_delta: float,
        particles_mean: float,
        particles_std: float,
        weight_max: float,
        weight_entropy: float,
    ) -> None:
        """Append small consensus stats to disk to avoid holding long histories in RAM."""
        stats_log_path = self._get_stats_log_path()
        stats_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not stats_log_path.exists()
        with stats_log_path.open("a") as f:
            if write_header:
                f.write(
                    "iteration,temperature,consensus_to_best,consensus_delta,"
                    "particles_to_consensus_mean,particles_to_consensus_std,"
                    "max_weight,weight_entropy\n"
                )
            f.write(
                f"{iteration},{temperature},{consensus_to_best},{consensus_delta},"
                f"{particles_mean},{particles_std},"
                f"{weight_max},{weight_entropy}\n"
            )

    def _persist_loss_stats(
        self,
        iteration: int,
        best_cost: float,
        mean_cost: float,
    ) -> None:
        """Append rollout cost stats to disk every few iterations."""
        loss_log_path = self._get_loss_log_path()
        loss_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not loss_log_path.exists()
        with loss_log_path.open("a") as f:
            if write_header:
                f.write("iteration,best_cost,mean_cost\n")
            f.write(f"{iteration},{best_cost},{mean_cost}\n")

    def _persist_cost_stats(
        self,
        iteration: int,
        best_cost: float,
        mean_cost: float,
        best_ever_cost: float,
    ) -> None:
        cost_log_path = self._get_cost_log_path()
        if cost_log_path is None:
            return
        cache_key = str(cost_log_path)
        last_iter = _LAST_COST_LOG_ITER_BY_PATH.get(cache_key)
        if last_iter == iteration:
            return
        cost_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not cost_log_path.exists()
        with cost_log_path.open("a") as f:
            if write_header:
                f.write("iteration,best_cost,mean_cost,best_ever_cost\n")
            f.write(f"{iteration},{best_cost},{mean_cost},{best_ever_cost}\n")
        _LAST_COST_LOG_ITER_BY_PATH[cache_key] = iteration

    def _persist_base_cost_stats(
        self,
        iteration: int,
        base_pos_cost: float,
        base_ori_cost: float,
        ori_weight: float,
        weighted_sum: float,
        unweighted_sum: float,
    ) -> None:
        base_log_path = self._get_base_cost_log_path()
        base_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not base_log_path.exists()
        with base_log_path.open("a") as f:
            if write_header:
                f.write(
                    "iteration,base_pos_cost,base_ori_cost,ori_weight,weighted_sum,unweighted_sum\n"
                )
            f.write(
                f"{iteration},{base_pos_cost},{base_ori_cost},"
                f"{ori_weight},{weighted_sum},{unweighted_sum}\n"
            )

    def _persist_terminal_cost_stats(
        self,
        iteration: int,
        base_pos_cost: float,
        base_ori_cost: float,
        foot_pos_cost: float,
        foot_ori_cost: float,
        base_height_cost: float,
        base_height: float,
        support_cost: float,
    ) -> None:
        term_log_path = self._get_terminal_cost_log_path()
        term_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not term_log_path.exists()
        with term_log_path.open("a") as f:
            if write_header:
                f.write(
                    "iteration,base_pos_cost,base_ori_cost,foot_pos_cost,foot_ori_cost,"
                    "base_height_cost,base_height,support_cost\n"
                )
            f.write(
                f"{iteration},{base_pos_cost},{base_ori_cost},"
                f"{foot_pos_cost},{foot_ori_cost},{base_height_cost},{base_height},{support_cost}\n"
            )

    def log_loaded_particles_stats(
        self,
        particles: jax.Array,
        state_mjx_data,
        tk: jax.Array | None,
    ) -> None:
        from algs.cbo_x import CustomCBOParams

        if not getattr(self.controller, "load_particles_from_disk", False):
            return
        if state_mjx_data is None or tk is None:
            return
        tq = jnp.linspace(tk[0], tk[-1], self.controller.ctrl_steps)
        controls = self.controller.interp_func(tq, tk, particles)
        states, rollouts = self.controller.eval_rollouts(
            self.controller.model, state_mjx_data, controls, particles
        )
        costs = rollouts.costs
        costs_2d = costs
        if costs.ndim == 1:
            costs_2d = costs[:, None]
        tmp_params = CustomCBOParams(
            tk=tk,
            mean=jnp.mean(particles, axis=0),
            rng=jax.random.PRNGKey(0),
            particles=particles,
            per_particle_consensus=jnp.broadcast_to(
                jnp.mean(particles, axis=0), particles.shape
            ),
            temperature=jnp.array(self.controller.initial_temperature),
            consensus_to_best=jnp.array(0.0),
            particles_to_consensus_mean=jnp.array(0.0),
            particles_to_consensus_std=jnp.array(0.0),
            weight_max=jnp.array(0.0),
            weight_entropy=jnp.array(0.0),
            iteration=jnp.array(0),
        )

        class _Rollouts:
            def __init__(self, costs, knots):
                self.costs = costs
                self.knots = knots

        tmp_rollouts = _Rollouts(costs=costs_2d, knots=particles)
        consensus, _, _ = self.controller.cal_consensus(tmp_rollouts, tmp_params)
        best_idx = int(np.asarray(jnp.argmin(jnp.sum(costs_2d, axis=1))))
        best_final_state = jax.tree.map(lambda x: x[best_idx, -1], states)
        best_terminal_cost = float(
            np.asarray(self.controller.task.terminal_cost(best_final_state))
        )
        consensus_knots = consensus[None, ...]
        consensus_controls = self.controller.interp_func(tq, tk, consensus_knots)
        consensus_states, _ = self.controller.eval_rollouts(
            self.controller.model, state_mjx_data, consensus_controls, consensus_knots
        )
        consensus_final_state = jax.tree.map(lambda x: x[0, -1], consensus_states)
        consensus_terminal_cost = float(
            np.asarray(self.controller.task.terminal_cost(consensus_final_state))
        )
        persisted_consensus = None
        try:
            particles_dir = self.controller.io.resolve_particles_dir()
            consensus_path = particles_dir / "consensus_latest.npy"
            if consensus_path.exists():
                persisted_consensus = np.load(consensus_path)
        except FileNotFoundError:
            persisted_consensus = None
        best_unweighted_cost = None
        if hasattr(self.controller.task, "terminal_cost_unweighted"):
            best_unweighted_cost = float(
                np.asarray(self.controller.task.terminal_cost_unweighted(best_final_state))
            )
        consensus_unweighted_cost = None
        if hasattr(self.controller.task, "terminal_cost_unweighted"):
            consensus_unweighted_cost = float(
                np.asarray(self.controller.task.terminal_cost_unweighted(consensus_final_state))
            )
        consensus_np = np.asarray(consensus_knots)
        log_line(
            f"consensus point re-calculated from loaded particles: {consensus_np}",
            self._get_last_output_path(),
        )
        if persisted_consensus is not None:
            log_line(
                f"persisted consensus point (consensus_latest.npy): {persisted_consensus}",
                self._get_last_output_path(),
            )
            if persisted_consensus.ndim == 2:
                persisted_knots = jnp.asarray(persisted_consensus)[None, ...]
                persisted_controls = self.controller.interp_func(tq, tk, persisted_knots)
                persisted_states, _ = self.controller.eval_rollouts(
                    self.controller.model,
                    state_mjx_data,
                    persisted_controls,
                    persisted_knots,
                )
                persisted_final_state = jax.tree.map(
                    lambda x: x[0, -1], persisted_states
                )
                persisted_terminal_cost = float(
                    np.asarray(self.controller.task.terminal_cost(persisted_final_state))
                )
                persisted_unweighted_cost = None
                if hasattr(self.controller.task, "terminal_cost_unweighted"):
                    persisted_unweighted_cost = float(
                        np.asarray(
                            self.controller.task.terminal_cost_unweighted(
                                persisted_final_state
                            )
                        )
                    )
                if persisted_unweighted_cost is None:
                    log_line(
                        f"persisted consensus terminal cost: {persisted_terminal_cost:.6f}",
                        self._get_last_output_path(),
                    )
                else:
                    log_line(
                        f"persisted consensus terminal cost: {persisted_terminal_cost:.6f}; "
                        f"unweighted (full configuration): {persisted_unweighted_cost:.6f}",
                        self._get_last_output_path(),
                    )
                self.controller._prev_consensus = jnp.asarray(persisted_consensus)
        if best_unweighted_cost is None:
            log_line(
                f"best particle terminal cost: {best_terminal_cost:.6f}",
                self._get_last_output_path(),
            )
        else:
            log_line(
                f"best particle terminal cost: {best_terminal_cost:.6f}; "
                f"unweighted (full configuration): {best_unweighted_cost:.6f}",
                self._get_last_output_path(),
            )
        if consensus_unweighted_cost is None:
            log_line(
                f"consensus terminal cost: {consensus_terminal_cost:.6f}",
                self._get_last_output_path(),
            )
        else:
            log_line(
                f"consensus terminal cost: {consensus_terminal_cost:.6f}; "
                f"unweighted (full configuration): {consensus_unweighted_cost:.6f}",
                self._get_last_output_path(),
            )

    def log_iteration(
        self,
        iteration: int,
        policy_params,
        rollouts,
        best_cost: float,
        mean_cost: float,
        state_mjx_data=None,
    ) -> None:
        current_temp = float(np.asarray(policy_params.temperature))
        consensus_to_best = float(np.asarray(policy_params.consensus_to_best))
        particles_mean = float(np.asarray(policy_params.particles_to_consensus_mean))
        particles_std = float(np.asarray(policy_params.particles_to_consensus_std))
        weight_max = float(np.asarray(policy_params.weight_max))
        weight_entropy = float(np.asarray(policy_params.weight_entropy))
        consensus_delta = 0.0
        prev = getattr(self.controller, "_prev_consensus", None)
        if prev is not None:
            consensus_delta = float(
                np.asarray(jnp.linalg.norm(policy_params.mean - prev))
            )
        self.controller._prev_consensus = jnp.asarray(policy_params.mean)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = getattr(self.controller, "experiment_start_time", None)
        if isinstance(start_time, datetime):
            start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            time_prefix = f"time: {now_str}; start: {start_str}; "
        else:
            time_prefix = f"time: {now_str}; "
        best_ever_cost, best_ever_iter = update_best_ever_cost(
            self.controller, best_cost, iteration
        )
        log_line(
            f"{time_prefix}current temperautre: {current_temp}; "
            f"consensus->best: {consensus_to_best:.4f}; "
            f"consensus_delta: {consensus_delta:.4f}; "
            f"particles->consensus mean/std: {particles_mean:.4f}/{particles_std:.4f}; "
            f"best/mean/best_ever cost: {best_cost:.6f}/{mean_cost:.6f}/{best_ever_cost:.6f}; "
            f"best_ever iter: {best_ever_iter}; "
            f"max weight: {weight_max:.4f}; entropy: {weight_entropy:.4f}",
            self._get_last_output_path(),
        )

        base_cost_values = None
        terminal_cost_values = None
        if state_mjx_data is not None and hasattr(self.controller.task, "base_cost_terms"):
            try:
                tk = policy_params.tk
                knots = policy_params.mean[None, ...]
                tq = jnp.linspace(tk[0], tk[-1], self.controller.ctrl_steps)
                controls = self.controller.interp_func(tq, tk, knots)[0]
                controls = controls[None, ...]
                states, _ = self.controller.eval_rollouts(
                    self.controller.model, state_mjx_data, controls, knots
                )
                final_state = jax.tree.map(lambda x: x[0, -1], states)
                (
                    base_pos_cost,
                    base_ori_cost,
                    ori_weight,
                    weighted_sum,
                    unweighted_sum,
                ) = self.controller.task.base_cost_terms(final_state)
                if getattr(self.controller.task, "use_se3_twist", False) and hasattr(
                    self.controller.task, "_se3_twist_calc"
                ):
                    q_ref = self.controller.task._get_reference_configuration(
                        final_state.time
                    )
                    twist = self.controller.task._se3_twist_calc.from_qpos(
                        final_state.qpos, q_ref
                    )
                    se3_norm = float(np.asarray(jnp.linalg.norm(twist)))
                    log_line(
                        f"se3 twist norm: {se3_norm:.6f}",
                        self._get_last_output_path(),
                    )
                log_line(
                    "base pos cost: "
                    f"{float(np.asarray(base_pos_cost)):.6f}; "
                    "base ori cost: "
                    f"{float(np.asarray(base_ori_cost)):.6f}; "
                    "ori weight: "
                    f"{float(np.asarray(ori_weight)):.6f}; "
                    "weighted sum: "
                    f"{float(np.asarray(weighted_sum)):.6f}; "
                    "unweighted sum: "
                    f"{float(np.asarray(unweighted_sum)):.6f}",
                    self._get_last_output_path(),
                )
                base_cost_values = (
                    float(np.asarray(base_pos_cost)),
                    float(np.asarray(base_ori_cost)),
                    float(np.asarray(ori_weight)),
                    float(np.asarray(weighted_sum)),
                    float(np.asarray(unweighted_sum)),
                )
                if hasattr(self.controller.task, "terminal_cost_controller") and self.controller.task.terminal_cost_controller is not None:
                    weights = self.controller.task.terminal_cost_controller.weights
                    log_line(
                        "terminal weights: "
                        f"base height loss weight={weights.base_height_weight:.4f} "
                        f"support loss weight={weights.support_weight:.4f}",
                        self._get_last_output_path(),
                    )
                if hasattr(self.controller.task, "terminal_cost"):
                    terminal_cost = float(
                        np.asarray(self.controller.task.terminal_cost(final_state))
                    )
                    unweighted_cost = None
                    if hasattr(self.controller.task, "terminal_cost_unweighted"):
                        unweighted_cost = float(
                            np.asarray(
                                self.controller.task.terminal_cost_unweighted(final_state)
                            )
                        )
                    if unweighted_cost is None:
                        log_line(
                            f"terminal cost: {terminal_cost:.6f}",
                            self._get_last_output_path(),
                        )
                    else:
                        log_line(
                            f"terminal cost: {terminal_cost:.6f}; "
                            f"unweighted (full configuration): {unweighted_cost:.6f}",
                            self._get_last_output_path(),
                        )
                if hasattr(self.controller.task, "_get_foot_position_errors") and hasattr(
                    self.controller.task, "_get_foot_orientation_errors"
                ):
                    left_pos_err, right_pos_err = self.controller.task._get_foot_position_errors(
                        final_state
                    )
                    left_ori_err, right_ori_err = self.controller.task._get_foot_orientation_errors(
                        final_state
                    )
                    foot_pos_cost = float(
                        np.asarray(
                            jnp.sum(jnp.square(left_pos_err))
                            + jnp.sum(jnp.square(right_pos_err))
                        )
                    )
                    foot_ori_cost = float(
                        np.asarray(
                            jnp.sum(jnp.square(left_ori_err))
                            + jnp.sum(jnp.square(right_ori_err))
                        )
                    )
                    base_height_cost = 0.0
                    base_height = 0.0
                    support_cost = 0.0
                    is_stable = None
                    if hasattr(self.controller.task, "_get_reference_configuration") and hasattr(
                        final_state, "time"
                    ):
                        q_ref = self.controller.task._get_reference_configuration(
                            final_state.time
                        )
                        qpos = final_state.qpos
                        base_height = float(np.asarray(qpos[2]))
                        base_height_cost = float(
                            np.asarray(jnp.square(qpos[2] - q_ref[2]))
                        )
                        if hasattr(self.controller.task, "_get_foot_positions"):
                            left_pos, right_pos = self.controller.task._get_foot_positions(
                                final_state
                            )
                            support_cost = float(
                                np.asarray(
                                    support_polygon_cost(
                                        qpos[:2], jnp.stack([left_pos, right_pos])
                                    )
                                )
                            )
                            is_stable = bool(
                                np.asarray(
                                    is_standing_stable(
                                        qpos,
                                        final_state.qvel,
                                        q_ref,
                                        base_pos_cost_tol=self.controller.task.stand_base_pos_cost_tol,
                                        base_ori_cost_tol=self.controller.task.stand_base_ori_cost_tol,
                                        min_base_height=self.controller.task.stand_min_base_height,
                                        max_base_lin_vel=self.controller.task.stand_max_base_lin_vel,
                                        max_base_ang_vel=self.controller.task.stand_max_base_ang_vel,
                                        foot_positions=jnp.stack([left_pos, right_pos]),
                                        support_margin=self.controller.task.stand_support_margin,
                                        joint_range=self.controller.task.model.jnt_range
                                        if self.controller.task.stand_joint_limit_violation is not None
                                        else None,
                                        max_joint_limit_violation=self.controller.task.stand_joint_limit_violation,
                                    )
                                )
                            )
                    if is_stable is False:
                        log_line(
                            "unstable: "
                            f"base_height={base_height:.4f} "
                            f"target_height={float(np.asarray(q_ref[2])):.4f} "
                            f"support_cost={support_cost:.6f}\n",
                            self._get_last_output_path(),
                        )
                    terminal_cost_values = (
                        float(np.asarray(base_pos_cost)),
                        float(np.asarray(base_ori_cost)),
                        foot_pos_cost,
                        foot_ori_cost,
                        base_height_cost,
                        base_height,
                        support_cost,
                    )
            except Exception as exc:
                log_line(f"base cost logging failed: {exc}", self._get_last_output_path())

        prev_best = best_ever_cost

        persist_every = getattr(self.controller, "persist_every", 1)
        if iteration % persist_every == 0:
            self._persist_consensus_stats(
                iteration,
                current_temp,
                consensus_to_best,
                consensus_delta,
                particles_mean,
                particles_std,
                weight_max,
                weight_entropy,
            )
            self._persist_loss_stats(
                iteration,
                best_cost,
                mean_cost,
            )
            self._persist_cost_stats(
                iteration,
                best_cost,
                mean_cost,
                prev_best,
            )
            if base_cost_values is not None:
                self._persist_base_cost_stats(
                    iteration,
                    *base_cost_values,
                )
            if terminal_cost_values is not None:
                self._persist_terminal_cost_stats(
                    iteration,
                    *terminal_cost_values,
                )
