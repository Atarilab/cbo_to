from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from algs.cbo_io import CBOxIO
from algs.cbo_stats import CBOxStatsRecorder


@dataclass
class CustomCBOParams(SamplingParams):
    """Policy parameters for model-predictive path integral control.
    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """
    particles: jax.Array
    per_particle_consensus: jax.Array
    temperature: jax.Array
    consensus_to_best: jax.Array
    particles_to_consensus_mean: jax.Array
    particles_to_consensus_std: jax.Array
    weight_max: jax.Array
    weight_entropy: jax.Array
    iteration: jax.Array


class CBOx(SamplingBasedController):
    """CustomCBO
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        lambda_: float = 0.5,
        time_step: float = 0.005,
        flag_anistropic: bool = True,
        flag_multiply_cost: bool = False,
        flag_multiply_knot_weights: bool = False,
        limit_temp=None,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.initial_temperature = temperature
        self.lambda_ = lambda_   # decay
        self.sigma = noise_level
        self.time_step = time_step
        self.flag_anistropic = flag_anistropic
        self.flag_multiply_cost = flag_multiply_cost
        self.flag_multiply_knot_weights = flag_multiply_knot_weights
        self.limit_temp = limit_temp
        self.persist_every = 100
        self.persist_particles_every = 500
        self.persist_particles_latest: int | None = 20
        self.load_particles_from_disk = False
        self.load_particles_dir = None
        self.load_particles_algorithm: str | None = None
        self.load_particles_iteration: int | None = None
        self.reset_temperature_on_load = False
        self.fill_random_particles_on_load = False
        self.io = CBOxIO(self)
        self.stats = CBOxStatsRecorder(self)

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> CustomCBOParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)

        particles, loaded_temperature, rng = self.io.load_particles_from_disk(
            _params.tk, _params.rng, _params.mean
        )
        if particles is None:
            print("randomly generating initial particles")
            """Sample a control sequence."""
            rng, sample_rng = jax.random.split(_params.rng)
            # FIXME:
            # Here I split twice to match the weird split twice behavior
            # in evosax.py for CMA-ES
            rng, sample_rng = jax.random.split(rng)
            print(sample_rng)
            noise = jax.random.normal(
                sample_rng,
                (
                    self.num_samples,
                    self.num_knots,
                    self.task.model.nu  # dof
                ),
            )
            particles = _params.mean + \
                noise * self.noise_level
        else:
            if rng is None:
                rng = _params.rng
            _params = _params.replace(mean=jnp.mean(particles, axis=0))
        initial_temperature = self.initial_temperature
        if loaded_temperature is not None and not self.reset_temperature_on_load:
            initial_temperature = loaded_temperature

        return CustomCBOParams(tk=_params.tk,
                               mean=_params.mean,
                               rng=rng,
                               particles=particles,
                               per_particle_consensus=jnp.broadcast_to(_params.mean, particles.shape),
                               temperature=jnp.array(initial_temperature),
                               consensus_to_best=jnp.array(0.0),
                               particles_to_consensus_mean=jnp.array(0.0),
                               particles_to_consensus_std=jnp.array(0.0),
                               weight_max=jnp.array(0.0),
                               weight_entropy=jnp.array(0.0),
                               iteration=jnp.array(0))

    def sample_knots(self, params: CustomCBOParams
                     ) -> Tuple[jax.Array, CustomCBOParams]:
        """
        Sample a control sequence. Different from other sampling based
        algorithms, sampling only has to be done once for CBO
        """
        return params.particles, params

    def cal_consensus(
        self, rollouts: Trajectory, params: CustomCBOParams, *, iteration: int | None = None
    ) -> jax.Array:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # costs.shape (N,)
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        # params.particles  # FIXME: add particle norm as regularization
        temperature = params.temperature
        weights = jax.nn.softmax(-costs / temperature, axis=0)
        # weights.shape = (N,)
        weight_max = jnp.max(weights)
        weight_entropy = -jnp.sum(weights * jnp.log(jnp.clip(weights, 1e-12, 1.0)))
        temp_decay = getattr(self, "temp_decay", 1.0)
        decay_temp = temperature / temp_decay
        if self.limit_temp is not None:
            new_temperature = jnp.where(decay_temp < self.limit_temp, self.limit_temp, decay_temp)
        else:
            new_temperature = decay_temp
        params = params.replace(temperature=new_temperature)
        # print(f"current temperautre: {self.temperature}")
        # if self.temperature < 1e-16:
        #     self.temperature = 1e-16
        consensus = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        best_idx = jnp.argmin(costs)
        best_knots = rollouts.knots[best_idx]
        consensus_to_best = jnp.linalg.norm(consensus - best_knots)
        particle_dists = jnp.linalg.norm(params.particles - consensus, axis=(1, 2))
        particles_mean = jnp.mean(particle_dists)
        particles_std = jnp.std(particle_dists)
        params = params.replace(
            consensus_to_best=consensus_to_best,
            particles_to_consensus_mean=particles_mean,
            particles_to_consensus_std=particles_std,
            weight_max=weight_max,
            weight_entropy=weight_entropy,
        )
        return consensus, costs, params

    def update_params(self,
                      params: CustomCBOParams,
                      rollouts: Trajectory) -> CustomCBOParams:
        """
        Update the mean with an exponentially weighted average,
        comes after self.sample_knots:

            def _optimize_scan_body(params: Any, _: Any):¬
        147             # Sample random control sequences from spline knots¬
        148             knots, params = self.sample_knots(params)¬
        """
        arr_consensus, costs, params = self.cal_consensus(
            rollouts, params, iteration=getattr(self, "_iteration_for_stats", None)
        )
        params_new_consensus = params.replace(mean=arr_consensus)
        params = self.update_particles(params_new_consensus, costs)
        return params

    def update_particles(self, params: CustomCBOParams, costs) -> jax.Array:
        """one SDE step"""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_knots,
                self.task.model.nu,
            ),
        )  # noise should be the safe for all particles
        particles_directions = params.particles - params.mean
        # n_p x n_k x (dof)
        n_k = particles_directions.shape[1]

        particles_directions_norm = jnp.linalg.norm(
            particles_directions, axis=(-2, -1), keepdims=True
        )
        anistropic_noise = jnp.multiply(particles_directions, noise)
        isotropic_noise = jnp.multiply(particles_directions_norm, noise)
        # probability weights over knots (heavier on earlier knots)
        knot_weights = jnp.linspace(1.0, 0.2, n_k)
        knot_weights = knot_weights / knot_weights.sum()
        cost_min = costs.min()
        cost_max = costs.max()
        norm_costs = (costs - cost_min) / (cost_max - cost_min + 1e-8)
        if self.flag_anistropic:
            final_noise = anistropic_noise
        else:
            final_noise = isotropic_noise
        if self.flag_multiply_cost:
            final_noise *= norm_costs[:, None, None]
        if self.flag_multiply_knot_weights:
            final_noise *= knot_weights[None, :, None]
        particles = params.particles  \
            - self.time_step * self.lambda_ * particles_directions \
            + jnp.sqrt(self.time_step) * self.sigma * final_noise

        return params.replace(particles=particles)
