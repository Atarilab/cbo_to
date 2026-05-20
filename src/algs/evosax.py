from typing import Any, Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

# from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from algs.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from algs.cbo_io import CBOxIO

from evosax.algorithms.base import EvolutionaryAlgorithm

# Generic types for evosax
EvoParams = Any
EvoState = Any


@dataclass
class EvosaxParams(SamplingParams):
    """Policy parameters for evosax optimizers.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
        opt_state: The state of the evosax optimizer (covariance, etc.).
    """

    opt_state: EvoState


class Evosax(SamplingBasedController):
    """A generic controller that allows us to use any evosax optimizer.

    See https://github.com/RobertTLange/evosax/ for details and a list of
    available optimizers.
    """

    def __init__(
        self,
        task: Task,
        optimizer: EvolutionaryAlgorithm,
        num_samples: int,
        es_params: EvoParams = None,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        **kwargs,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            optimizer: The evosax optimizer to use.
            num_samples: The number of control tapes to sample.
            es_params: The parameters for the evosax optimizer.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
            **kwargs: Additional keyword arguments for the optimizer.
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

        self.strategy = optimizer(
            population_size=num_samples,
            # Only to inform the dimension to evosax
            solution=jnp.zeros(task.model.nu * self.num_knots),
            **kwargs,
        )

        if es_params is None:
            es_params = self.strategy.default_params
        self.es_params = es_params
        self.num_samples = num_samples
        self.persist_particles_every = 500
        self.persist_particles_latest: int | None = None
        self.load_particles_dir = None
        self.load_particles_algorithm: str | None = None
        self.load_particles_iteration: int | None = None
        self.io = CBOxIO(self)

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> EvosaxParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        rng, init_rng = jax.random.split(_params.rng)

        mean = jnp.reshape(
            _params.mean, (self.task.model.nu * self.num_knots)
        )
        try:
            # Distribution-based algs (e.g., CMA-ES) expect mean.
            opt_state = self.strategy.init(
                key=init_rng, mean=mean, params=self.es_params
            )
        except TypeError:
            try:
                # Some strategies (e.g., SV_* variants) expect means for each population.
                num_pops = getattr(self.strategy, "num_populations", 1)
                means = jnp.tile(mean[None, :], (num_pops, 1))
                opt_state = self.strategy.init(
                    key=init_rng, means=means, params=self.es_params
                )
            except TypeError:
                try:
                    # Population-based algs may require an initial population + fitness.
                    pop_size = self.strategy.population_size
                    population = jnp.tile(mean[None, :], (pop_size, 1))
                    fitness = jnp.zeros((pop_size,))
                    opt_state = self.strategy.init(
                        key=init_rng,
                        population=population,
                        fitness=fitness,
                        params=self.es_params,
                    )
                except TypeError:
                    # Some population-based algs don't take mean/population in init.
                    opt_state = self.strategy.init(
                        key=init_rng, params=self.es_params
                    )
        return EvosaxParams(
            tk=_params.tk, mean=_params.mean, opt_state=opt_state, rng=rng
        )

    def sample_knots(
        self, params: EvosaxParams
    ) -> Tuple[jax.Array, EvosaxParams]:
        """Sample control sequences from the proposal distribution."""
        rng, sample_rng = jax.random.split(params.rng)
        jax.debug.print("sample_rng: {}", sample_rng)
        x, opt_state = self.strategy.ask(
            sample_rng, params.opt_state, self.es_params
        )

        # evosax works with vectors of decision variables, so we reshape U to
        # [batch_size, num_knots, nu].
        controls = jnp.reshape(
            x,
            (
                self.strategy.population_size,
                self.num_knots,
                self.task.model.nu,
            ),
        )

        return controls, params.replace(opt_state=opt_state, rng=rng)

    def update_params(
        self, params: EvosaxParams, rollouts: Trajectory
    ) -> EvosaxParams:
        """Update the policy parameters based on the rollouts."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        x = jnp.reshape(rollouts.knots, (self.strategy.population_size, -1))

        # Evosax 0.2 requires a key in its tell() methods
        rng, update_rng = jax.random.split(params.rng)

        opt_state, _ = self.strategy.tell(
            key=update_rng, population=x, fitness=costs, state=params.opt_state, params=self.es_params
        )

        best_idx = jnp.argmin(costs)
        best_knots = rollouts.knots[best_idx]

        # By default, opt_state stores the best member ever, rather than the
        # best member from the current generation. We want to just use the best
        # member from this generation, since the cost landscape is constantly
        # changing.
        # Ensure shapes match opt_state fields to keep scan carry types stable.
        if hasattr(opt_state, "best_solution") and hasattr(opt_state, "best_fitness"):
            opt_state = opt_state.replace(
                best_solution=jnp.reshape(x[best_idx], opt_state.best_solution.shape),
                best_fitness=jnp.reshape(costs[best_idx], opt_state.best_fitness.shape),
            )
            # original:
            # best_solution=x[best_idx], best_fitness=costs[best_idx]

        if hasattr(opt_state, "mean"):
            mean_vec = opt_state.mean
        elif hasattr(opt_state, "population"):
            mean_vec = jnp.mean(opt_state.population, axis=0)
        else:
            mean_vec = jnp.reshape(best_knots, (-1,))

        mean = jnp.reshape(
            mean_vec,
            (
                self.num_knots,
                self.task.model.nu,
            ),
        )

        return params.replace(mean=mean, opt_state=opt_state, rng=rng)
