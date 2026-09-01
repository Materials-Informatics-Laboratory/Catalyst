"""Catalyst public package namespace.

The top-level module stays lightweight for packaging/metadata inspection, while
convenience attributes are imported lazily.  Import failures are intentionally
not swallowed: requesting a broken public component should expose the real
exception rather than silently omitting it from the namespace.
"""

from ._version import __version__

__all__ = [
    "__version__",
    "Catalyst",
    "CatalystParameterError",
    "GNN",
    "GNNTask",
    "VectorChannelAdapter",
    "GraphMultiScalarAdapter",
    "build_task_model",
    "validate_task_batch",
    "task_from_parameters",
    "build_model",
]


def __getattr__(name):
    if name in {"Catalyst", "CatalystParameterError"}:
        from . import observer
        return getattr(observer, name)

    if name in {
        "GNN",
        "GNNTask",
        "VectorChannelAdapter",
        "GraphMultiScalarAdapter",
        "build_task_model",
        "validate_task_batch",
        "task_from_parameters",
        "build_model",
    }:
        from . import ml
        from .ml import gnn
        return getattr(gnn, name)

    raise AttributeError(f"module 'catalyst' has no attribute {name!r}")
