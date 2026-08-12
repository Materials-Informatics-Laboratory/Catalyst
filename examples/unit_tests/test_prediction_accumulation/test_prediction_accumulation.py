"""Regression tests for prediction selection, target resolution, and accumulation."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from catalyst.ml.gnn.GNN import GNN
from catalyst.ml.gnn.modules.utils.predict import accumulate_predictions


def test_none_optional_prediction_settings_do_not_override_accumulator_defaults():
    gnn = GNN(nn.Identity(), torch.device("cpu"))
    kwargs = gnn._prediction_kwargs(
        {
            "model_dict": {
                "loss_params": {},
                "prediction_params": {
                    "channel_mode": None,
                    "normalize_by": None,
                    "output_key": None,
                    "target_key": "target_scalar",
                },
            }
        }
    )
    assert "channel_mode" not in kwargs
    assert "normalize_by" not in kwargs
    assert kwargs["target_key"] == "target_scalar"


def test_direct_graph_scalar_target_is_reshaped_safely():
    data = SimpleNamespace(target_scalar=torch.tensor([[1.0], [2.0]]))
    pred = torch.tensor([1.1, 1.9])
    preds, target, vec = accumulate_predictions(
        pred,
        data,
        loss_tag="exact",
        target_key="target_scalar",
    )
    assert tuple(preds.shape) == (2,)
    assert tuple(target.shape) == (2,)
    assert not vec


def test_direct_graph_multiscalar_preserves_channels_without_vector_semantics():
    data = SimpleNamespace(target_scalars=torch.arange(6, dtype=torch.float32))
    pred = torch.zeros((2, 3), dtype=torch.float32)
    preds, target, vec = accumulate_predictions(
        pred,
        data,
        loss_tag="exact",
        target_key="target_scalars",
        prefer_equivariant_key="scalar",
    )
    assert tuple(preds.shape) == (2, 3)
    assert tuple(target.shape) == (2, 3)
    # Compatibility flag is true for any multichannel tensor, but the task layer
    # is what distinguishes independent scalars from a geometric vector.
    assert vec


def test_direct_node_scalar_is_not_graph_pooled():
    data = SimpleNamespace(target_scalar=torch.tensor([0.2, 0.4, 0.8]))
    pred = torch.tensor([0.1, 0.5, 0.7])
    preds, target, vec = accumulate_predictions(
        pred,
        data,
        loss_tag="node",
        target_key="target_scalar",
    )
    assert torch.equal(preds, pred)
    assert torch.equal(target, data.target_scalar)
    assert not vec


def test_direct_vector_and_gradient_targets_keep_n_by_3_shape():
    target = torch.randn(5, 3)
    data = SimpleNamespace(target_vector=target)

    vector_pred = torch.randn(5, 3)
    preds, y, vec = accumulate_predictions(
        {"vector": vector_pred},
        data,
        loss_tag="node",
        output_key="vector",
        target_key="target_vector",
    )
    assert preds.shape == y.shape == (5, 3)
    assert vec

    gradient_pred = torch.randn(5, 3)
    preds, y, vec = accumulate_predictions(
        {"scalar": torch.randn(5, 1), "gradient": gradient_pred},
        data,
        loss_tag="node",
        output_key="gradient",
        target_key="target_vector",
    )
    assert preds.shape == y.shape == (5, 3)
    assert vec


def test_legacy_list_output_defaults_to_target_channel_mode():
    # Two graph samples with two primary nodes each and one secondary entity
    # each. One output channel should aggregate to one scalar per graph.
    data = SimpleNamespace(
        node_G_batch=torch.tensor([0, 0, 1, 1]),
        node_A_batch=torch.tensor([0, 1]),
        y=torch.tensor([1.0, 2.0]),
    )
    pred = [
        torch.ones((4, 1)),
        torch.ones((2, 1)),
    ]
    preds, target, vec = accumulate_predictions(
        pred,
        data,
        loss_tag="exact",
        return_y=True,
    )
    assert preds.numel() == 2
    assert target.numel() == 2
    assert not vec
