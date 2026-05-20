from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from mujoco.mjx._src.math import quat_sub


@dataclass
class SE3TwistCalculator:
    """Compute a 6D se(3) twist using the SE(3) log map."""

    def from_errors(self, base_pos_err: jnp.ndarray, base_ori_err: jnp.ndarray) -> jnp.ndarray:
        """
        Return twist [vx, vy, vz, wx, wy, wz] from translation and SO(3) log errors.

        The translation is mapped with V^{-1}(phi) to be consistent with SE(3) log.
        """
        base_pos_err = jnp.asarray(base_pos_err).reshape(-1)[:3]
        # phi is the SO(3) log-map rotation vector (axis * angle).
        base_ori_err = jnp.asarray(base_ori_err).reshape(-1)[:3]
        rho = self._se3_translation_log(base_pos_err, base_ori_err)
        return jnp.concatenate([rho, base_ori_err], axis=0)

    def from_qpos(self, qpos: jnp.ndarray, q_ref: jnp.ndarray) -> jnp.ndarray:
        """
        Return twist from qpos and reference qpos (pos + quaternion).

        Uses translation difference in world frame and SO(3) log error from quat_sub.
        """
        base_pos_err = qpos[:3] - q_ref[:3]
        # phi: SO(3) log-map rotation vector from quaternion difference.
        base_ori_err = quat_sub(qpos[3:7], q_ref[3:7])
        return self.from_errors(base_pos_err, base_ori_err)

    def _se3_translation_log(self, t: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
        """
        phi is the rotation vector:
            base_ori_err = quat_sub(qpos[3:7], q_ref[3:7])
        theta is its magnitude (rotation angle).

        Compute V^{-1}(phi) * t for SE(3) log map.

        The SE(3) exponential map uses:
          T = [ R, V(phi) * rho; 0, 1 ]
        where R = Exp(phi), rho is the translation part of the twist, and
        V(phi) accounts for the coupling between rotation and translation.
        Note: V(phi) is a nonlinear function of phi (depends on sin/cos).
        This function applies V^{-1}(phi) to recover rho from the translation t.
        """
        theta = jnp.linalg.norm(phi)
        eps = 1e-8
        phi_hat = self._skew(phi)
        phi_hat_sq = phi_hat @ phi_hat
        theta2 = theta * theta
        theta3 = theta2 * theta

        def small_angle() -> jnp.ndarray:
            # 2nd-order Taylor of V^{-1} around phi ≈ 0.
            return jnp.eye(3) - 0.5 * phi_hat + (1.0 / 12.0) * phi_hat_sq

        def general_angle() -> jnp.ndarray:
            # Closed-form V^{-1} for non-zero rotation.
            a = 0.5
            b = (1.0 / theta2) - (1.0 + jnp.cos(theta)) / (2.0 * theta * jnp.sin(theta))
            return jnp.eye(3) - a * phi_hat + b * phi_hat_sq

        v_inv = jnp.where(theta < eps, small_angle(), general_angle())
        return v_inv @ t

    def _skew(self, v: jnp.ndarray) -> jnp.ndarray:
        vx, vy, vz = v[0], v[1], v[2]
        return jnp.array(
            [
                [0.0, -vz, vy],
                [vz, 0.0, -vx],
                [-vy, vx, 0.0],
            ]
        )
