"""Encoder exports."""

from .atomic_encoders import (
    AtomicGraphEncoder,
    GenericFeatureEncoder,
    Encoder_atomic,
    Encoder_generic,
    attach_input_order_aliases,
    backfill_hidden_aliases,
)

try:
    from .equivariant_encoders import (
        EquivariantAtomicEncoder,
        EquivariantEncoder,
        AtomicEquivariantEncoder,
    )
except Exception:
    pass

__all__ = [
    "AtomicGraphEncoder",
    "GenericFeatureEncoder",
    "Encoder_atomic",
    "Encoder_generic",
    "attach_input_order_aliases",
    "backfill_hidden_aliases",
]

for _name in ("EquivariantAtomicEncoder", "EquivariantEncoder", "AtomicEquivariantEncoder"):
    if _name in globals():
        __all__.append(_name)
