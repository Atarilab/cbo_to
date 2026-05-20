from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.tree_util as jtu
import numpy as np


class ParamsIO:
    """Persist algorithm Params to disk (call from non-jitted Python code)."""

    def __init__(self, controller) -> None:
        self.controller = controller
        self._params_dir = Path("tmp/persisted_params")

    def _resolve_persist_config(self) -> tuple[int | None, int | None]:
        persist_every = getattr(self.controller, "persist_params_every", None)
        if persist_every is None:
            persist_every = getattr(self.controller, "persist_particles_every", 500)
        persist_latest = getattr(self.controller, "persist_params_latest", None)
        if persist_latest is None:
            persist_latest = getattr(self.controller, "persist_particles_latest", None)
        return persist_every, persist_latest

    def _normalize_persist_latest(self, persist_latest: int | bool | None) -> int | None:
        if isinstance(persist_latest, bool):
            persist_latest = 1 if persist_latest else None
        if persist_latest is not None and persist_latest <= 0:
            persist_latest = None
        return persist_latest

    def persist_params(self, params: Any, iteration: int) -> tuple[bool, bool]:
        persist_every, persist_latest = self._resolve_persist_config()
        persist_latest = self._normalize_persist_latest(persist_latest)

        should_persist_latest = (
            persist_latest is not None and iteration % int(persist_latest) == 0
        )
        should_persist_every = (
            persist_every is not None and iteration % persist_every == 0
        )
        if not (should_persist_latest or should_persist_every):
            return False, False

        self._params_dir.mkdir(parents=True, exist_ok=True)
        params_host = jax.device_get(params)
        def _safe_array(value):
            try:
                return np.asarray(value)
            except TypeError as exc:
                if "PRNGKey" in str(exc) or "prng_key" in str(exc):
                    return np.asarray(jax.random.key_data(value))
                raise

        params_np = jtu.tree_map(_safe_array, params_host)

        if should_persist_latest:
            np.save(self._params_dir / "params_latest.npy", params_np, allow_pickle=True)
            (self._params_dir / "params_latest_iter.txt").write_text(
                f"{iteration}\n", encoding="utf-8"
            )

        if should_persist_every:
            file_path = self._params_dir / f"params_iter_{iteration:05d}.npy"
            np.save(file_path, params_np, allow_pickle=True)
        return should_persist_latest, should_persist_every
