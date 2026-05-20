from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TerminalCostWeights:
    base_pos_weight: float
    base_ori_weight: float
    foot_pos_weight: float
    foot_ori_weight: float
    base_height_weight: float
    support_weight: float


@dataclass
class TerminalCostScheduler:
    k: float
    update_every: int
    mode: str
    min_weight: float
    max_weight: float
    target_base_pos_cost: float
    target_base_ori_cost: float
    target_foot_pos_cost: float
    target_foot_ori_cost: float
    target_base_height_cost: float
    use_height_factor: bool = True
    target_support_cost: float = 0.0
    cost_log_path: Path | None = None

    def set_cost_log_path(self, path: Path | str | None) -> None:
        if path is None:
            self.cost_log_path = None
        else:
            self.cost_log_path = Path(path)

    def update_weights(self, iteration: int, weights: TerminalCostWeights) -> TerminalCostWeights:
        if self.update_every <= 0 or iteration % self.update_every != 0:
            return weights
        if self.cost_log_path is None or not self.cost_log_path.exists():
            return weights

        last_line = self._read_last_data_line(self.cost_log_path)
        if last_line is None:
            return weights

        parts = last_line.split(",")
        if len(parts) < 8:
            return weights
        try:
            base_pos_cost = float(parts[1])
            base_ori_cost = float(parts[2])
            foot_pos_cost = float(parts[3])
            foot_ori_cost = float(parts[4])
            base_height_cost = float(parts[5])
            height_factor = float(parts[6])
            support_cost = float(parts[7])
        except ValueError:
            return weights

        def update(weight: float, cost: float, target: float) -> float:
            delta = self.k * (cost - target)
            if self.use_height_factor:
                delta *= height_factor
            if self.mode == "multiplicative":
                new_weight = weight * (1.0 + delta)
            else:
                new_weight = weight + delta
            if new_weight < self.min_weight:
                return self.min_weight
            if new_weight > self.max_weight:
                return self.max_weight
            return new_weight

        return TerminalCostWeights(
            base_pos_weight=update(weights.base_pos_weight, base_pos_cost, self.target_base_pos_cost),
            base_ori_weight=update(weights.base_ori_weight, base_ori_cost, self.target_base_ori_cost),
            foot_pos_weight=weights.foot_pos_weight,
            foot_ori_weight=weights.foot_ori_weight,
            base_height_weight=update(
                weights.base_height_weight, base_height_cost, self.target_base_height_cost
            ),
            support_weight=update(weights.support_weight, support_cost, self.target_support_cost),
        )

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


@dataclass
class TerminalCostController:
    weights: TerminalCostWeights
    scheduler: TerminalCostScheduler | None = None

    def set_cost_log_path(self, path: Path | str | None) -> None:
        if self.scheduler is not None:
            self.scheduler.set_cost_log_path(path)

    def update(self, iteration: int) -> None:
        if self.scheduler is None:
            return
        self.weights = self.scheduler.update_weights(iteration, self.weights)
