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


class Adam:
    """Adam over the fp32 master weights; re-quantizes the target after each step."""

    def __init__(self, target, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, clip=None):
        self.target = target
        self.lr = float(lr)
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.clip = None if clip is None else float(clip)
        if hasattr(target, "parameters"):
            self._params = target.parameters()
        else:
            self._params = list(target)
        self._requantize = getattr(target, "requantize", None)
        self._m = {}
        self._v = {}
        self._t = 0

    def _clip_grads(self):
        total = 0.0
        for p in self._params:
            total += float(np.sum(p.grad * p.grad))
        norm = total ** 0.5
        if norm > self.clip and norm > 0.0:
            scale = self.clip / norm
            for p in self._params:
                p.grad *= scale

    def step(self):
        if self.clip is not None:
            self._clip_grads()
        self._t += 1
        for p in self._params:
            g = p.grad
            m = self._m.get(id(p))
            v = self._v.get(id(p))
            if m is None:
                m = np.zeros_like(p.data)
                v = np.zeros_like(p.data)
            m = self.b1 * m + (1.0 - self.b1) * g
            v = self.b2 * v + (1.0 - self.b2) * (g * g)
            self._m[id(p)] = m
            self._v[id(p)] = v
            mhat = m / (1.0 - self.b1 ** self._t)
            vhat = v / (1.0 - self.b2 ** self._t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
        if callable(self._requantize):
            self._requantize()

    def zero_grad(self):
        for p in self._params:
            p.grad[:] = 0.0
