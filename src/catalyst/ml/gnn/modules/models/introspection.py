"""Public helpers for inspecting wrapped Catalyst GNN models."""

from __future__ import annotations

from typing import Any


_FEATURE_METHODS = ("forward_features", "decode_features")
_WRAPPER_ATTRIBUTES = ("module", "model")


def _supports_feature_api(model: Any) -> bool:
    return all(callable(getattr(model, name, None)) for name in _FEATURE_METHODS)


def unwrap_feature_model(model: Any):
    """Return the wrapped model that exposes Catalyst's feature API.

    Catalyst models may be wrapped by task adapters, DistributedDataParallel,
    or the high-level :class:`catalyst.ml.gnn.GNN` training object.  This helper
    follows the conventional ``module`` and ``model`` wrapper attributes until
    it finds an object implementing both ``forward_features`` and
    ``decode_features``.

    Parameters
    ----------
    model
        A Catalyst model or a supported wrapper around one.

    Returns
    -------
    object
        The first wrapped object exposing the feature API.

    Raises
    ------
    TypeError
        If no feature-capable model can be found.
    """
    current = model
    seen = set()
    chain = []

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(type(current).__name__)

        if _supports_feature_api(current):
            return current

        next_model = None
        for attr in _WRAPPER_ATTRIBUTES:
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current:
                next_model = candidate
                break

        current = next_model

    raise TypeError(
        "Could not find a wrapped Catalyst model implementing "
        "forward_features() and decode_features(). Traversed: "
        + " -> ".join(chain)
    )


__all__ = ["unwrap_feature_model"]
