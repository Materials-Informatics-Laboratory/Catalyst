"""Catalyst public package namespace."""

__version__ = "0.1.0-alpha"

# Keep the top-level namespace light and tolerant. Most users should import from
# subpackages directly, for example:
#
#     from catalyst.ml.gnn import GNNTask, build_task_model
#     from catalyst.observer import Catalyst
#
# The convenience imports below are guarded so importing ``catalyst`` does not
# fail during metadata-only operations or partial optional-dependency installs.

__all__ = ["__version__"]

try:
    from .observer import Catalyst
except Exception:
    pass
else:
    __all__.append("Catalyst")

try:
    from .ml.gnn import (
        GNN,
        GNNTask,
        VectorChannelAdapter,
        build_task_model,
        validate_task_batch,
        task_from_parameters,
        build_model,
    )
except Exception:
    pass
else:
    __all__.extend(
        [
            "GNN",
            "GNNTask",
            "VectorChannelAdapter",
            "build_task_model",
            "validate_task_batch",
            "task_from_parameters",
            "build_model",
        ]
    )
