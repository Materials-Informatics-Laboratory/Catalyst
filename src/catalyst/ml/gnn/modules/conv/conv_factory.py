"""
Convolution factory for Catalyst GNN modules.

Recommended location:
    catalyst/ml/gnn/modules/conv/factory.py

This file does not define new convolution math. It only maps string names such
as "mesh", "gated_gcn", "pna", etc. to the actual conv classes.

The conv classes themselves stay in files like:
    mgn_conv.py
    gated_gcn.py
    gated_gcn_v2.py
    gine_conv.py
    edge_conditioned_conv.py
    pna_conv.py
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Iterable, Optional

from torch import nn


# =============================================================================
# Activation builder
# =============================================================================


def build_activation(act=None):
    """
    Normalize activation specifications.

    Accepts:
        None
        nn.Module instance
        nn.Module class
        strings: "silu", "swish", "relu", "gelu", "tanh", "leaky_relu"
    """
    if act is None:
        return nn.SiLU()

    if isinstance(act, nn.Module):
        return act

    if isinstance(act, type) and issubclass(act, nn.Module):
        return act()

    if isinstance(act, str):
        key = act.lower().strip()

        if key in {"silu", "swish"}:
            return nn.SiLU()
        if key == "relu":
            return nn.ReLU()
        if key == "gelu":
            return nn.GELU()
        if key == "tanh":
            return nn.Tanh()
        if key in {"leaky_relu", "lrelu"}:
            return nn.LeakyReLU()

    raise ValueError(f"Unsupported activation specification: {act!r}")


# =============================================================================
# Registry and aliases
# =============================================================================


CONV_ALIASES: Dict[str, str] = {
    "mesh": "mesh",
    "mgn": "mesh",
    "mesh_graph_nets": "mesh",
    "meshgraphnets": "mesh",

    "gcn": "gated_gcn",
    "gated_gcn": "gated_gcn",
    "gatedgcn": "gated_gcn",

    "gated_gcn_v2": "gated_gcn_v2",
    "gatedgcn_v2": "gated_gcn_v2",
    "gcn_v2": "gated_gcn_v2",

    "gine": "gine",
    "gin": "gine",

    "edge_conditioned": "edge_conditioned",
    "edgeconditioned": "edge_conditioned",
    "nnconv": "edge_conditioned",
    "ecc": "edge_conditioned",

    "pna": "pna",
}


# Optional user registry for external/custom convs.
CONV_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def normalize_conv_type(conv_type: str) -> str:
    key = str(conv_type).lower().strip()

    if key in CONV_REGISTRY:
        return key

    if key not in CONV_ALIASES:
        supported = sorted(set(CONV_ALIASES) | set(CONV_REGISTRY))
        raise ValueError(
            f"Unsupported conv_type={conv_type!r}. "
            f"Supported options are: {supported}"
        )

    return CONV_ALIASES[key]


def register_conv(name: str, factory: Callable[..., nn.Module], *aliases: str) -> None:
    """
    Register a custom conv factory.

    Example:
        register_conv("my_conv", MyConv, "mine", "custom")
    """
    name = str(name).lower().strip()
    CONV_REGISTRY[name] = factory

    for alias in aliases:
        CONV_ALIASES[str(alias).lower().strip()] = name


# =============================================================================
# Lazy class import helpers
# =============================================================================


def _import_class_from_candidates(
    class_name: str,
    module_candidates: Iterable[str],
):
    """
    Import a class from likely module locations.

    This avoids requiring every conv file to have a perfectly standardized file
    name immediately. It first tries direct modules, then falls back to the conv
    package __init__.py exports.
    """
    package = __package__  # e.g. catalyst.ml.gnn.modules.conv

    for module_name in module_candidates:
        try:
            module = importlib.import_module(f"{package}.{module_name}")
        except ImportError:
            continue

        if hasattr(module, class_name):
            return getattr(module, class_name)

    # Fallback to package-level export from conv/__init__.py.
    try:
        package_module = importlib.import_module(package)
    except ImportError:
        package_module = None

    if package_module is not None and hasattr(package_module, class_name):
        return getattr(package_module, class_name)

    raise ImportError(
        f"Could not import {class_name}. Tried modules {list(module_candidates)} "
        f"and package-level export from {package}."
    )


def _get_mesh_conv():
    return _import_class_from_candidates(
        "MeshGraphNetsConv",
        ["mgn_conv", "mesh_graph_nets_conv", "meshgraphnets_conv"],
    )


def _get_gated_gcn():
    return _import_class_from_candidates(
        "GatedGCN",
        ["gated_gcn", "gcn_conv", "gated_gcn_conv"],
    )


def _get_gated_gcn_v2():
    return _import_class_from_candidates(
        "GatedGCN_v2",
        ["gated_gcn_v2", "gcn_conv", "gated_gcn_conv"],
    )


def _get_gine_conv():
    return _import_class_from_candidates(
        "GINEConv",
        ["gine_conv", "gine"],
    )


def _get_edge_conditioned_conv():
    return _import_class_from_candidates(
        "EdgeConditionedConv",
        ["edge_conditioned_conv", "edge_conditioned", "nnconv"],
    )


def _get_pna_conv():
    return _import_class_from_candidates(
        "PNAConv",
        ["pna_conv", "pna"],
    )


# =============================================================================
# Public builder
# =============================================================================


def build_conv_layer(
    conv_type: str,
    node_dim: int,
    edge_dim: int,
    aggr_scheme: str = "add",
    act=None,
    **kwargs: Any,
) -> nn.Module:
    """
    Build one conv layer from a string.

    Standard interface expected by processors:
        x_new, edge_attr_new = conv(x, edge_index, edge_attr)
    """
    conv_type = normalize_conv_type(conv_type)
    act = build_activation(act)

    if conv_type in CONV_REGISTRY:
        return CONV_REGISTRY[conv_type](
            node_dim,
            edge_dim,
            aggr_scheme=aggr_scheme,
            act=act,
            **kwargs,
        )

    if conv_type == "mesh":
        cls = _get_mesh_conv()
        # Existing MeshGraphNetsConv may not accept act.
        return cls(
            node_dim,
            edge_dim,
            aggr_scheme=aggr_scheme,
            **kwargs,
        )

    if conv_type == "gated_gcn":
        cls = _get_gated_gcn()
        return cls(
            node_dim,
            edge_dim,
            aggr_scheme=aggr_scheme,
            act=act,
            **kwargs,
        )

    if conv_type == "gated_gcn_v2":
        cls = _get_gated_gcn_v2()
        return cls(
            node_dim,
            edge_dim,
            aggr_scheme=aggr_scheme,
            act=act,
            **kwargs,
        )

    if conv_type == "gine":
        cls = _get_gine_conv()
        return cls(
            node_dim,
            edge_dim,
            aggr_scheme=aggr_scheme,
            act=act,
            **kwargs,
        )

    if conv_type == "edge_conditioned":
        cls = _get_edge_conditioned_conv()
        return cls(
            node_dim,
            edge_dim,
            aggr_scheme=aggr_scheme,
            act=act,
            **kwargs,
        )

    if conv_type == "pna":
        cls = _get_pna_conv()
        return cls(
            node_dim,
            edge_dim,
            aggr_scheme=aggr_scheme,
            act=act,
            **kwargs,
        )

    raise RuntimeError(f"Internal conv factory error for conv_type={conv_type!r}.")


__all__ = [
    "CONV_ALIASES",
    "CONV_REGISTRY",
    "build_activation",
    "normalize_conv_type",
    "register_conv",
    "build_conv_layer",
]
