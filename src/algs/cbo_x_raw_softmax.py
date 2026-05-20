from __future__ import annotations

import jax
import jax.numpy as jnp

from algs.cbo_x import CBOx, CustomCBOParams
from hydrax.alg_base import Trajectory


class CBOxRaw(CBOx):
    """CBOx variant with a manual softmax in consensus calculation."""

    def _softmax_raw(self, logits: jax.Array, axis: int = 0) -> jax.Array:
        max_logits = jnp.max(logits, axis=axis, keepdims=True)
        shifted = logits - max_logits
        exp_vals = jnp.exp(shifted)
        denom = jnp.sum(exp_vals, axis=axis, keepdims=True)
        return exp_vals / (denom + 1e-12)

    def cal_consensus(
        self, rollouts: Trajectory, params: CustomCBOParams
    ) -> jax.Array:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        params.particles  # FIXME: add particle norm as regularization
        temperature = params.temperature
        scaled_costs = -costs / temperature
        weights = self._softmax_raw(scaled_costs, axis=0)
        weight_max = jnp.max(weights)
        weight_entropy = -jnp.sum(weights * jnp.log(jnp.clip(weights, 1e-12, 1.0)))
        temp_decay = getattr(self, "temp_decay", 1.0)
        decay_temp = temperature / temp_decay
        if self.limit_temp is not None:
            new_temperature = jnp.where(decay_temp < self.limit_temp, self.limit_temp, decay_temp)
        else:
            new_temperature = decay_temp
        params = params.replace(temperature=new_temperature)
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
