"""Processor exports."""

from .order_processor import OrderProcessor
from .order_processor import Processor as OrderProcessorAlias
from .scalar_processor import ScalarProcessor
from .scalar_processor import Processor as ScalarProcessorAlias

from .equivariant_processor import EquivariantProcessor
from .equivariant_processor import Processor as EquivariantProcessorAlias

__all__ = [
    "OrderProcessor",
    "ScalarProcessor",
    "OrderProcessorAlias",
    "ScalarProcessorAlias",
]

__all__.extend(["EquivariantProcessor", "EquivariantProcessorAlias"])
