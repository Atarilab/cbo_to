from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseOrientationWeightScheduler:
    target_base_ori_cost: float
    k: float
    min_weight: float
    max_weight: float
    update_every: int
    mode: str = "additive"
    use_base_pos_cost: bool = True
    base_pos_target: float = 0.01
    pre_activation_weight: float = 1.0
    base_cost_log_path: Path | None = None
    _activated: bool = False

    def set_cost_log_path(self, path: Path | str | None) -> None:
        if path is None:
            self.base_cost_log_path = None
        else:
            self.base_cost_log_path = Path(path)

    def update(self, iteration: int, current_weight: float) -> float:
        if self.update_every <= 0 or iteration % self.update_every != 0:
            return current_weight
        if self.base_cost_log_path is None or \
                not self.base_cost_log_path.exists():
            return current_weight

        last_line = self._read_last_data_line(self.base_cost_log_path)
        if last_line is None:
            return current_weight

        parts = last_line.split(",")
        if len(parts) < 3:
            return current_weight
        try:
            base_pos_cost = float(parts[1])
            base_ori_cost = float(parts[2])
        except ValueError:
            return current_weight

        if not self._activated:
            if base_pos_cost > self.base_pos_target:
                return self.pre_activation_weight
            # equivalent else
            self._activated = True

        # in case weight scheduling is activated but position cost fails to be
        # good enough, use original weight
        if base_pos_cost > self.base_pos_target:
            return self.pre_activation_weight

        base_pos_factor = base_pos_cost if self.use_base_pos_cost else 1.0
        delta = self.k * (base_ori_cost - self.target_base_ori_cost) \
            * base_pos_factor

        if self.mode == "multiplicative":
            new_weight = current_weight * (1.0 + delta)
        else:
            new_weight = current_weight + delta

        if new_weight < self.min_weight:
            return self.min_weight
        if new_weight > self.max_weight:
            return self.max_weight
        return new_weight

    def _read_last_data_line(self, path: Path) -> str | None:
        try:
            with path.open("rb") as f:
                f.seek(0, 2)
                end = f.tell()
                if end == 0:
                    return None
                pos = end - 1
                while pos >= 0:
                    f.seek(pos)
                    if f.read(1) == b"\n" and pos != end - 1:
                        break
                    pos -= 1
                f.seek(pos + 1)
                line = f.read().decode("utf-8", errors="ignore").strip()
        except OSError:
            return None
        if not line or line.startswith("iteration,"):
            return None
        return line
