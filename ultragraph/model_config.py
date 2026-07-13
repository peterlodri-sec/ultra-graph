"""Declarative model configuration via ``@ug.model``.

Define a model class with type-annotated fields and defaults — the decorator
builds the corresponding ``GPT`` instance automatically.

Example::

    @ug.model
    class TinyLM:
        vocab: int = 256
        d_model: int = 64
        n_layers: int = 2
        n_heads: int = 2

    model = TinyLM()         # -> GPT(vocab=256, d_model=64, n_layers=2, n_heads=2)
    model.fit(x, y, 10, 0.01)

For custom model classes, set ``klass`` (default ``GPT``)::

    @ug.model(klass=Mesh)
    class MoEModel:
        vocab: int = 256
        d_model: int = 128
        n_layers: int = 4
        n_heads: int = 4
        n_experts: int = 4

Every field in the config class becomes a keyword argument to the constructor.
"""
from __future__ import annotations

import dataclasses
from typing import Any, TypeVar

T = TypeVar("T")


def model(cls_or_klass: Any = None, *, klass: type = None):
    """Decorator to create a declarative model config class.

    When ``cls_or_klass`` is a class, it wraps it as a model factory.
    When ``klass`` is provided, use that as the target model class.

    The decorated class's fields become constructor kwargs; calling the config
    class builds the model.
    """
    if cls_or_klass is not None and isinstance(cls_or_klass, type):
        return _build_config_class(cls_or_klass, klass)
    # Called as @ug.model(klass=X) — cls_or_klass is None or a non-class
    return lambda c: _build_config_class(c, cls_or_klass if cls_or_klass is not None else klass)


def _build_config_class(config_cls: type, target_klass: type | None = None) -> type:
    """Convert a declarative config class into a model factory."""
    from .model import GPT

    target = target_klass or GPT

    # Convert mutable defaults to default_factory for dataclass compatibility
    annotations = getattr(config_cls, "__annotations__", {})
    for name, typ in annotations.items():
        default = getattr(config_cls, name, dataclasses.MISSING)
        if default is not dataclasses.MISSING and not isinstance(default, type):
            if isinstance(default, (list, dict, set)):
                setattr(config_cls, name, dataclasses.field(default_factory=lambda d=default: type(d)(d)))

    cls = dataclasses.dataclass(config_cls)
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        fields = {f.name for f in dataclasses.fields(cls)}
        for k in kwargs:
            if k not in fields:
                raise TypeError(f"{cls.__name__} has no field {k!r}")
        original_init(self, *args, **kwargs)

    def build(self):
        """Build and return the model instance."""
        fields = dataclasses.fields(self)
        kwargs = {}
        for f in fields:
            val = getattr(self, f.name)
            if isinstance(val, dataclasses.Field):
                continue
            kwargs[f.name] = val
        return target(**kwargs)

    def __call__(self, *args, **kwargs):
        """Calling the config instance builds the model."""
        return self.build()

    cls.__init__ = __init__
    cls.build = build
    cls.__call__ = __call__
    cls._model_class = target
    return cls
