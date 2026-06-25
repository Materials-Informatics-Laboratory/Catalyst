"""
ScalarProcessor for Catalyst GNN modules.

Recommended location:
    catalyst/ml/gnn/modules/processors/scalar_processor.py

This is the simple one-graph scalar/invariant processor. It updates only the
primary node/edge graph:

    h_1, h_2 = conv(h_1, edge_index_2, h_2)

It does not use h_3 or edge_index_3.

Use this for standard scalar GNNs such as GINE/PNA/NNConv/GatedGCN/MGN when you
do not want the ALIGNN-style angle/line-graph update.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from torch import nn

from ..conv.conv_factory import build_activation, build_conv_layer


# =============================================================================
# Alias helpers
# =============================================================================


def _has_attr(data, name: str) -> bool:
    return hasattr(data, name) and getattr(data, name) is not None


def _first_attr(data, *names: str, required: bool = False):
    for name in names:
        if _has_attr(data, name):
            return getattr(data, name)

    if required:
        raise AttributeError(f"Could not find any of these graph attributes: {names}")

    return None


def attach_scalar_aliases(data):
    """
    Attach scalar one-graph aliases.

    Preferred:
        h_1, h_2, edge_index_2

    Legacy atomic:
        h_atm, h_bnd, edge_index_G

    Legacy generic:
        h_g_node, h_a_node, edge_index_G

    Basic PyG-like:
        h_node, h_edge, edge_index
    """
    if not _has_attr(data, "h_1"):
        value = _first_attr(data, "h_atm", "h_g_node", "h_node")
        if value is not None:
            data.h_1 = value

    if not _has_attr(data, "h_2"):
        value = _first_attr(data, "h_bnd", "h_a_node", "h_edge")
        if value is not None:
            data.h_2 = value

    if not _has_attr(data, "edge_index_2"):
        value = _first_attr(data, "edge_index_G", "edge_index")
        if value is not None:
            data.edge_index_2 = value

    return data


def backfill_scalar_aliases(data):
    """
    Backfill older Catalyst hidden names from h_1/h_2.

    This lets old decoders/readouts continue to work while the internal processor
    uses the order-style h_1/h_2 names.
    """
    if _has_attr(data, "h_1"):
        if _has_attr(data, "x_atm"):
            data.h_atm = data.h_1
        if _has_attr(data, "node_G"):
            data.h_g_node = data.h_1
        if _has_attr(data, "x"):
            data.h_node = data.h_1

    if _has_attr(data, "h_2"):
        if _has_attr(data, "x_bnd"):
            data.h_bnd = data.h_2
        if _has_attr(data, "node_A"):
            data.h_a_node = data.h_2
        if _has_attr(data, "edge_attr"):
            data.h_edge = data.h_2

    return data


# =============================================================================
# Processor
# =============================================================================


class ScalarProcessor(nn.Module):
    """
    Simple scalar/invariant one-graph message-passing processor.

    Parameters
    ----------
    dim
        Hidden feature dimension.
    num_convs
        Number of message-passing layers.
    conv_type
        Conv type from catalyst.ml.gnn.modules.conv.factory.
    aggr_scheme
        Aggregation scheme passed to the conv layers.
    act
        Activation specification.
    conv_kwargs
        Extra kwargs passed to each conv layer.
    """

    def __init__(
        self,
        dim: int,
        num_convs: int,
        conv_type: str = "mesh",
        aggr_scheme: str = "add",
        act=None,
        conv_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.dim = dim
        self.num_convs = num_convs
        self.conv_type = conv_type
        self.aggr_scheme = aggr_scheme
        self.act = build_activation(act)

        conv_kwargs = dict(conv_kwargs or {})

        self.convs = nn.ModuleList(
            [
                build_conv_layer(
                    conv_type=conv_type,
                    node_dim=dim,
                    edge_dim=dim,
                    aggr_scheme=aggr_scheme,
                    act=self.act,
                    **conv_kwargs,
                )
                for _ in range(num_convs)
            ]
        )

    def _validate_required_fields(self, data) -> None:
        missing = []

        if not _has_attr(data, "h_1"):
            missing.append("h_1 or h_atm/h_g_node/h_node")

        if not _has_attr(data, "h_2"):
            missing.append("h_2 or h_bnd/h_a_node/h_edge")

        if not _has_attr(data, "edge_index_2"):
            missing.append("edge_index_2 or edge_index_G/edge_index")

        if missing:
            raise AttributeError(
                "ScalarProcessor is missing required graph fields after encoding: "
                + ", ".join(missing)
            )

    def forward(self, data):
        data = attach_scalar_aliases(data)
        self._validate_required_fields(data)

        for conv in self.convs:
            data.h_1, data.h_2 = conv(
                data.h_1,
                data.edge_index_2,
                data.h_2,
            )

        return backfill_scalar_aliases(data)


# Compatibility alias if a user wants Processor to mean the simple scalar processor
# inside this file.
Processor = ScalarProcessor


__all__ = [
    "ScalarProcessor",
    "Processor",
    "attach_scalar_aliases",
    "backfill_scalar_aliases",
]
