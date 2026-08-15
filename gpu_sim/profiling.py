"""Low-overhead, opt-in timing for device-resident simulation stages."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

import torch


class TensorStageProfiler:
    """Measure stage durations with CUDA synchronization only when enabled."""

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self._totals: dict[str, float] = defaultdict(float)
        self._calls: dict[str, int] = defaultdict(int)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.synchronize()
        start = perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self._totals[name] += perf_counter() - start
            self._calls[name] += 1

    def as_dict(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "elapsed_seconds": round(total, 6),
                "calls": self._calls[name],
                "average_seconds": round(total / self._calls[name], 9),
            }
            for name, total in self._totals.items()
        }
