"""Machine-learning entry points."""

from .training import run_training, setup_training
from .inference import run_inference, setup_inference
from .gnn import *
from .gnn import __all__ as _gnn_all
from .nn import *
from .nn import __all__ as _nn_all

__all__ = [
    "run_training",
    "setup_training",
    "run_inference",
    "setup_inference",
    *_gnn_all,
    *_nn_all,
]
