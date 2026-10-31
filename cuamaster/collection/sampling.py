"""Coverage-first and EMA frontier task sampling."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Hashable, Iterable


@dataclass
class TaskStats:
    ema_success_rate: float
    count: int = 0


class FrontierSampler:
    def __init__(
        self,
        *,
        target_success_rate: float = 0.5,
        ema_alpha: float = 0.2,
        sigma: float = 0.4,
        random_sample_ratio: float = 0.3,
        exploration_bonus: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.target_success_rate = target_success_rate
        self.ema_alpha = ema_alpha
        self.sigma = sigma
        self.random_sample_ratio = random_sample_ratio
        self.exploration_bonus = exploration_bonus
        self.stats: dict[Hashable, TaskStats] = {}
        self.random = random.Random(seed)

    def update(self, task_id: Hashable, success: float) -> None:
        value = float(success)
        if task_id not in self.stats:
            self.stats[task_id] = TaskStats(value, 1)
            return
        stats = self.stats[task_id]
        stats.ema_success_rate = (
            (1 - self.ema_alpha) * stats.ema_success_rate + self.ema_alpha * value
        )
        stats.count += 1

    def weight(self, task_id: Hashable) -> float:
        stats = self.stats.get(task_id)
        if stats is None:
            return 1.0 + self.exploration_bonus
        distance = stats.ema_success_rate - self.target_success_rate
        frontier = math.exp(-(distance * distance) / (2 * self.sigma * self.sigma))
        return frontier + self.exploration_bonus / math.sqrt(stats.count + 1)

    def sample(self, task_ids: Iterable[Hashable], count: int) -> list[Hashable]:
        candidates = list(task_ids)
        if count < 0:
            raise ValueError("count must be >= 0")
        if not candidates or count == 0:
            return []
        selected: list[Hashable] = []
        for _ in range(count):
            if self.random.random() < self.random_sample_ratio:
                selected.append(self.random.choice(candidates))
            else:
                selected.append(self.random.choices(
                    candidates,
                    weights=[self.weight(task_id) for task_id in candidates],
                    k=1,
                )[0])
        return selected

    def state_dict(self) -> dict:
        return {
            "target_success_rate": self.target_success_rate,
            "ema_alpha": self.ema_alpha,
            "sigma": self.sigma,
            "random_sample_ratio": self.random_sample_ratio,
            "exploration_bonus": self.exploration_bonus,
            "task_stats": {
                str(key): {
                    "ema_success_rate": value.ema_success_rate,
                    "count": value.count,
                }
                for key, value in self.stats.items()
            },
        }
