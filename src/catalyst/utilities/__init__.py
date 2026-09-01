"""General Catalyst utilities.

Ranking helpers are imported lazily to avoid a circular dependency between
catalyst.properties.structure_properties and catalyst.utilities.
"""

from .data_tools import *
from .distributions import *
from .sampling import *

_RANKING_EXPORTS = {
    "organize_rankings_atomic",
    "organize_rankings_generic",
}

__all__ = [name for name in globals() if not name.startswith("_")]
__all__ += sorted(_RANKING_EXPORTS)


def __getattr__(name):
    if name in _RANKING_EXPORTS:
        from . import rankings as _rankings
        return getattr(_rankings, name)

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )