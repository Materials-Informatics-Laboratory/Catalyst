"""Machine-learning entry points."""

try:
    from .training import run_training, run_active_learning, setup_training
except Exception:
    pass

try:
    from .inference import run_inference, setup_inference
except Exception:
    pass

try:
    from .gnn import *
except Exception:
    pass

try:
    from .nn import *
except Exception:
    pass

__all__ = [name for name in globals() if not name.startswith("_")]
