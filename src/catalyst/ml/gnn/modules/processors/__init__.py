"""Processor exports."""

from .order_processor import OrderProcessor
from .order_processor import Processor as OrderProcessorAlias
from .scalar_processor import ScalarProcessor
from .scalar_processor import Processor as ScalarProcessorAlias

try:
    from .equivariant_processor import EquivariantProcessor
    from .equivariant_processor import Processor as EquivariantProcessorAlias
except Exception:
    pass

__all__ = [
    "OrderProcessor",
    "ScalarProcessor",
    "OrderProcessorAlias",
    "ScalarProcessorAlias",
]

for _name in ("EquivariantProcessor", "EquivariantProcessorAlias"):
    if _name in globals():
        __all__.append(_name)
