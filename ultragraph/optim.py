"""Optimizers operating on the full-precision master weights in the ad-hoc store.

After a step, dense trees are re-quantized so the byte buffers reflect the updated
master weights (train fp32, deploy ternary).
"""
from __future__ import annotations

import numpy as np


class SGD:
    """Vanilla SGD with optional momentum.

    ``target`` may be anything exposing ``parameters() -> list[Tensor]`` (and,
    optionally, ``requantize()``), e.g. an ``UltraGraph``.
    """

    def __init__(self, target, lr: float = 0.1, momentum: float = 0.0, clip: float | None = None):
        self.target = target
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.clip = None if clip is None else float(clip)
        if hasattr(target, "parameters"):
            self._params = target.parameters()
        else:
            self._params = list(target)
        self._requantize = getattr(target, "requantize", None)
        self._vel: dict[int, np.ndarray] = {}

    def _clip_grads(self) -> None:
        total = 0.0
        for p in self._params:
            total += float(np.sum(p.grad * p.grad))
        norm = total ** 0.5
        if norm > self.clip and norm > 0.0:
            scale = self.clip / norm
            for p in self._params:
                p.grad *= scale

    def step(self) -> None:
        if self.clip is not None:
            self._clip_grads()
        for p in self._params:
            g = p.grad
            if self.momentum:
                v = self._vel.get(id(p))
                if v is None:
                    v = np.zeros_like(p.data)
                v = self.momentum * v + g
                self._vel[id(p)] = v
                p.data -= self.lr * v
            else:
                p.data -= self.lr * g
        if callable(self._requantize):
            self._requantize()

    def zero_grad(self) -> None:
        for p in self._params:
            p.grad[:] = 0.0
