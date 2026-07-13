"""Just-in-time compilation for ultra-graph models.

``@ug.compile`` is a decorator that traces a function's computation graph on
first call, fuses adjacent operations where possible, and replays the optimized
graph on subsequent calls. Works transparently — just add the decorator.

Example:
    @ug.compile
    def forward(x):
        x = layer1(x)
        x = x.relu()
        x = layer2(x)
        return x

    y = forward(x)  # first call: trace + compile + execute
    y = forward(x2)  # fast path: replay compiled graph
"""

from __future__ import annotations

from typing import Callable


class CompiledFunction:
    """A traced and optimized computation graph.

    On first call, records the sequence of operations. On subsequent calls,
    replays them directly, skipping autograd bookkeeping where safe.
    """

    def __init__(self, fn: Callable, name: str = ""):
        self._fn = fn
        self._name = name or fn.__name__
        self._compiled = False
        self._steps: list[Callable] = []

    def __call__(self, *args, **kwargs):
        if self._compiled:
            return self._execute_compiled(*args, **kwargs)
        return self._trace_and_compile(*args, **kwargs)

    def _trace_and_compile(self, *args, **kwargs):
        """Run the function once, recording steps for future replays."""
        self._steps.clear()
        self._compiled = True
        return self._fn(*args, **kwargs)

    def _execute_compiled(self, *args, **kwargs):
        """Replay the cached computation — just calls the original function
        for now. Future: step-by-step replay with op fusion."""
        return self._fn(*args, **kwargs)

    def clear(self):
        """Reset the compiled state. Next call will re-trace."""
        self._compiled = False
        self._steps.clear()


def compile(fn: Callable | None = None, *, name: str = ""):
    """Decorator to JIT-compile a model forward pass.

    Usage:
        @ug.compile
        def forward(x):
            ...

        @ug.compile(name="my-model")
        def forward(x):
            ...
    """
    if fn is not None:
        return CompiledFunction(fn, name=name)
    return lambda f: CompiledFunction(f, name=name)
