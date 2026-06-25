"""
OrderProcessor for Catalyst GNN modules.

Recommended location:
    catalyst/ml/gnn/modules/processors/order_processor.py

Processor versus conv
---------------------
A conv is one local message-passing update:

    x, edge_attr = conv(x, edge_index, edge_attr)

A processor is the architecture-level schedule that decides which convs to apply,
to which graph orders, and how many times.

This processor implements the ALIGNN-style 1/2/3-body update sequence:

    3-body / angular graph update:
        h_2, h_3 = conv(h_2, edge_index_3, h_3)

    2-body / atom-bond graph update:
        h_1, h_2 = conv(h_1, edge_index_2, h_2)

where:
    h_1 = atom or primary node hidden features
    h_2 = bond or primary edge hidden features
    h_3 = angle/dihedral or line-graph edge hidden features
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


def attach_order_input_aliases(data):
    """
    Attach order-style graph aliases from legacy Catalyst graph names.
    """
    if not _has_attr(data, "edge_index_2"):
        edge_index_2 = _first_attr(data, "edge_index_G")
        if edge_index_2 is not None:
            data.edge_index_2 = edge_index_2

    if not _has_attr(data, "edge_index_3"):
        edge_index_3 = _first_attr(data, "edge_index_A")
        if edge_index_3 is not None:
            data.edge_index_3 = edge_index_3

    return data


def attach_order_hidden_aliases(data):
    """
    Attach h_1/h_2/h_3 aliases from legacy hidden names.
    """
    if not _has_attr(data, "h_1"):
        h_1 = _first_attr(data, "h_atm", "h_g_node")
        if h_1 is not None:
            data.h_1 = h_1

    if not _has_attr(data, "h_2"):
        h_2 = _first_attr(data, "h_bnd", "h_a_node")
        if h_2 is not None:
            data.h_2 = h_2

    if not _has_attr(data, "h_3"):
        h_3 = _first_attr(data, "h_ang", "h_a_edge")
        if h_3 is not None:
            data.h_3 = h_3

    return data


def backfill_legacy_hidden_names(data):
    """
    Backfill old Catalyst hidden names from h_1/h_2/h_3.
    """
    if _has_attr(data, "h_1"):
        if _has_attr(data, "x_atm"):
            data.h_atm = data.h_1
        if _has_attr(data, "node_G"):
            data.h_g_node = data.h_1

    if _has_attr(data, "h_2"):
        if _has_attr(data, "x_bnd"):
            data.h_bnd = data.h_2
        if _has_attr(data, "node_A"):
            data.h_a_node = data.h_2

    if _has_attr(data, "h_3"):
        if _has_attr(data, "x_ang"):
            data.h_ang = data.h_3
        if _has_attr(data, "edge_A"):
            data.h_a_edge = data.h_3

    return data


# =============================================================================
# Processor
# =============================================================================


class OrderProcessor(nn.Module):
    """
    Generic 1/2/3-body processor.

    This is the modular replacement for the old ALIGNN Processor class.

    Parameters
    ----------
    dim
        Hidden feature dimension. The current encoders place h_1/h_2/h_3 in this
        same dimension.
    num_convs
        Number of repeated processor layers.
    conv_type
        Conv type for the 1/2-body update h_1,h_2.
    aggr_scheme
        Aggregation scheme passed to convs, usually "add", "mean", or "max"
        depending on conv support.
    encode_3body
        If True and h_3/edge_index_3 exist, update the angular/line graph before
        each 1/2-body update.
    threebody_conv_type
        Optional separate conv type for h_2,h_3 updates. If None, conv_type is
        reused.
    act
        Activation spec passed through conv.factory.build_activation.
    conv_kwargs
        Extra kwargs for 1/2-body convs.
    threebody_conv_kwargs
        Extra kwargs for 3-body convs.
    """

    def __init__(
        self,
        dim: int,
        num_convs: int,
        conv_type: str = "mesh",
        aggr_scheme: str = "add",
        encode_3body: bool = True,
        threebody_conv_type: Optional[str] = None,
        act=None,
        conv_kwargs: Optional[Dict[str, Any]] = None,
        threebody_conv_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.dim = dim
        self.num_convs = num_convs
        self.conv_type = conv_type
        self.threebody_conv_type = threebody_conv_type or conv_type
        self.aggr_scheme = aggr_scheme
        self.encode_3body = bool(encode_3body)
        self.act = build_activation(act)

        conv_kwargs = dict(conv_kwargs or {})
        threebody_conv_kwargs = dict(threebody_conv_kwargs or {})

        self.g_convs = nn.ModuleList(
            [
                build_conv_layer(
                    conv_type=self.conv_type,
                    node_dim=dim,
                    edge_dim=dim,
                    aggr_scheme=aggr_scheme,
                    act=self.act,
                    **conv_kwargs,
                )
                for _ in range(num_convs)
            ]
        )

        if self.encode_3body:
            self.a_convs = nn.ModuleList(
                [
                    build_conv_layer(
                        conv_type=self.threebody_conv_type,
                        node_dim=dim,
                        edge_dim=dim,
                        aggr_scheme=aggr_scheme,
                        act=self.act,
                        **threebody_conv_kwargs,
                    )
                    for _ in range(num_convs)
                ]
            )
        else:
            self.a_convs = nn.ModuleList()

    def _validate_required_fields(self, data) -> None:
        missing = []

        if not _has_attr(data, "h_1"):
            missing.append("h_1 or h_atm/h_g_node")

        if not _has_attr(data, "h_2"):
            missing.append("h_2 or h_bnd/h_a_node")

        if not _has_attr(data, "edge_index_2"):
            missing.append("edge_index_2 or edge_index_G")

        if missing:
            raise AttributeError(
                "OrderProcessor is missing required graph fields after encoding: "
                + ", ".join(missing)
            )

    def forward(self, data):
        data = attach_order_input_aliases(data)
        data = attach_order_hidden_aliases(data)

        self._validate_required_fields(data)

        has_3body = (
            self.encode_3body
            and len(self.a_convs) > 0
            and _has_attr(data, "h_3")
            and _has_attr(data, "edge_index_3")
        )

        for layer_idx in range(self.num_convs):
            if has_3body:
                data.h_2, data.h_3 = self.a_convs[layer_idx](
                    data.h_2,
                    data.edge_index_3,
                    data.h_3,
                )

            data.h_1, data.h_2 = self.g_convs[layer_idx](
                data.h_1,
                data.edge_index_2,
                data.h_2,
            )

        return backfill_legacy_hidden_names(data)


# Compatibility alias if you want to keep old ALIGNN imports alive:
Processor = OrderProcessor


__all__ = [
    "OrderProcessor",
    "Processor",
    "attach_order_input_aliases",
    "attach_order_hidden_aliases",
    "backfill_legacy_hidden_names",
]
