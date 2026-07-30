"""Decoder exports."""

from .standard_decoders import (
    ScalarDecoder,
    Decoder,
    PositiveScalarsDecoder,
    MultiScalarDecoder,
    PositiveKChannelDecoder,
    PositiveFeatureReadout,
)

try:
    from .equivariant_decoders import (
        EquivariantDecoder,
        GenericEquivariantDecoder,
        ScalarGradientDecoder,
    )
except Exception:
    pass

__all__ = [
    "ScalarDecoder",
    "Decoder",
    "PositiveScalarsDecoder",
    "MultiScalarDecoder",
    "PositiveKChannelDecoder",
    "PositiveFeatureReadout",
]

for _name in ("EquivariantDecoder", "GenericEquivariantDecoder", "ScalarGradientDecoder"):
    if _name in globals():
        __all__.append(_name)
