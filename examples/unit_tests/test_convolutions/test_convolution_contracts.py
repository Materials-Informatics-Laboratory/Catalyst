"""Forward/backward contract tests for every registered built-in convolution."""

from __future__ import annotations

import pytest
import torch

from catalyst.ml.gnn.modules.conv.conv_factory import build_conv_layer, normalize_conv_type


CANONICAL_CONVS = [
    "mesh",
    "gated_gcn",
    "gated_gcn_v2",
    "gine",
    "edge_conditioned",
    "pna",
]


@pytest.mark.parametrize("conv_type", CANONICAL_CONVS)
def test_builtin_convolution_forward_backward_contract(conv_type):
    torch.manual_seed(23)
    dim = 8
    x = torch.randn(5, dim, requires_grad=True)
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            [1, 4, 0, 2, 1, 3, 2, 4, 3, 0],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.randn(edge_index.size(1), dim, requires_grad=True)

    conv = build_conv_layer(
        conv_type,
        node_dim=dim,
        edge_dim=dim,
        aggr_scheme="add",
        avg_degree=2.0,
        use_degree_scalers=True,
    ) if conv_type == "pna" else build_conv_layer(
        conv_type,
        node_dim=dim,
        edge_dim=dim,
        aggr_scheme="add",
    )

    x_out, edge_out = conv(x, edge_index, edge_attr)
    assert x_out.shape == x.shape
    assert edge_out.shape == edge_attr.shape
    assert torch.isfinite(x_out).all()
    assert torch.isfinite(edge_out).all()

    loss = x_out.square().mean() + edge_out.square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert edge_attr.grad is not None and torch.isfinite(edge_attr.grad).all()


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("mgn", "mesh"),
        ("gcn", "gated_gcn"),
        ("gin", "gine"),
        ("nnconv", "edge_conditioned"),
    ],
)
def test_convolution_aliases_are_stable(alias, canonical):
    assert normalize_conv_type(alias) == canonical
