from __future__ import annotations

import jax
import jax.numpy as jnp

from algs.cbo_x import CBOx, CustomCBOParams
from pcbo.cal_consensus_polar_regularization import compute_polar_consensus
from pcbo.kernel import gaussian_kernel_neg_log
from hydrax.alg_base import Trajectory


class CBOxPolarized(CBOx):
    """CBOx variant with a manual softmax in consensus calculation."""

    def _softmax_raw(self, logits: jax.Array, axis: int = 0) -> jax.Array:
        max_logits = jnp.max(logits, axis=axis, keepdims=True)
        shifted = logits - max_logits
        exp_vals = jnp.exp(shifted)
        denom = jnp.sum(exp_vals, axis=axis, keepdims=True)
        return exp_vals / (denom + 1e-12)

    def _logsumexp_raw(self, values: jax.Array, axis: int = 0) -> jax.Array:
        max_vals = jnp.max(values, axis=axis, keepdims=True)
        shifted = values - max_vals
        return jnp.log(jnp.sum(jnp.exp(shifted), axis=axis, keepdims=True)) + max_vals

    def pairwise_dist(self, knots):
        arr_n_n_dist = self.kernel.neg_log(knots[None, :, :], knots[:, None, :])

    def compute_mean(self, costs, knots, ind: jax.Array | None = None) -> jax.Array:
        """Compute the mean m(x_i) of the particles."""
        if ind is None:
            ind = jnp.arange(self.num_samples)
        ind = jnp.asarray(ind)
        m_beta = jnp.zeros(self.x.shape)
        self.energy = costs[ind]
        for j in ind:
            neg_log_kernel = self.kernel.neg_log(knots[j, :], knots[ind, :])
            weights = -neg_log_kernel - self.beta * self.energy   # self.beta is regularization constant
            coeffs = jnp.exp(weights - self._logsumexp_raw(weights, axis=0))
            coeffs = jnp.expand_dims(coeffs, axis=1)
            m_beta = m_beta.at[j, :].set(jnp.sum(knots[ind, :] * coeffs, axis=0))
        return m_beta  # consensus
