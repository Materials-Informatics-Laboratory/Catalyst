"""
Processor exports for Catalyst GNN modules.
"""

from .order_processor import OrderProcessor
from .order_processor import Processor as OrderProcessorAlias

from .scalar_processor import ScalarProcessor
from .scalar_processor import Processor as ScalarProcessorAlias

__all__ = [
    "OrderProcessor",
    "ScalarProcessor",
    "OrderProcessorAlias",
    "ScalarProcessorAlias",
]
