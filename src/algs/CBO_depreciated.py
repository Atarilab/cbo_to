from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


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


class CustomCBO(SamplingBasedController):
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
        self.temperature = temperature
        self.lambda_ = 1.
        self.sigma = 1
        self.init_multiplier = 1.
        self.time_step = 0.1

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> CustomCBOParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)


        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(_params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )
        particles = _params.mean + self.init_multiplier * self.noise_level * noise  # Surely, this is not the best way to start

        return CustomCBOParams(tk=_params.tk, mean=_params.mean, rng=rng, particles=particles)

    def sample_knots(self, params: CustomCBOParams) -> Tuple[jax.Array, CustomCBOParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )
        cbo_direction = params.particles - params.mean
        # particles = params.particles  \
        #             - self.lambda_ * cbo_direction \
        #             + self.noise_level * noise
        particles = params.particles  \
                    - self.time_step * self.lambda_ * cbo_direction \
                    + self.time_step * self.sigma * self.noise_level * noise
                    # + self.time_step * self.sigma * jnp.multiply(cbo_direction, self.noise_level * noise)
        return particles, params.replace(rng=rng, particles=particles)

    def update_params(
        self, params: CustomCBOParams, rollouts: Trajectory
    ) -> CustomCBOParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        consensus = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        return params.replace(mean=consensus)


    # maybe directly write optimze again
