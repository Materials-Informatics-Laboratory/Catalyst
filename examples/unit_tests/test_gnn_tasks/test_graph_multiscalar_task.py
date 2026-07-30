"""Tests for independent graph-level multiscalar regression."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from catalyst.ml.gnn import (
    GNNTask,
    GraphMultiScalarAdapter,
    build_task_model,
    task_from_parameters,
)
from catalyst.ml.gnn.modules.decoders import MultiScalarDecoder
from catalyst.ml.gnn.modules.models.gnn_builder import build_default_decoder


class _IndexableNamespace(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)


class _OrderContributionModel(nn.Module):
    def forward(self, data):
        return [data.order_1_pred, data.order_2_pred]


class _DirectModel(nn.Module):
    def __init__(self, num_targets: int):
        super().__init__()
        self.num_targets = num_targets

    def forward(self, data):
        return torch.zeros((data.num_graphs, self.num_targets))


def test_graph_multiscalar_contract_and_parameter_roundtrip():
    task = GNNTask.graph_multiscalar(
        num_targets=3,
        target_key="y",
        target_names=["a", "b", "c"],
        normalize_by=None,
    )

    assert task.output_type == "scalar"
    assert task.output_level == "graph"
    assert task.out_dim == 3
    assert task.requires_graph_multiscalar_adapter
    assert not task.requires_vector_adapter

    parameters = {"model_dict": {"prediction_params": {}}}
    task.apply_to_catalyst_parameters(parameters)

    model_dict = parameters["model_dict"]
    prediction_params = model_dict["prediction_params"]
    assert model_dict["task"] == "graph_multiscalar"
    assert model_dict["task_out_dim"] == 3
    assert model_dict["task_target_names"] == ["a", "b", "c"]
    assert prediction_params["target_key"] == "y"
    assert prediction_params["channel_mode"] == "target"
    assert prediction_params["legacy_multichannel_shape"] is False
    assert prediction_params["normalize_by"] is None

    restored = task_from_parameters(parameters)
    assert restored == task


def test_graph_multiscalar_rejects_invalid_target_count():
    with pytest.raises(ValueError, match="num_targets >= 2"):
        GNNTask.graph_multiscalar(num_targets=1)

    with pytest.raises(ValueError, match="target_names"):
        GNNTask.graph_multiscalar(
            num_targets=3,
            target_names=["only", "two"],
        )


def test_multiscalar_decoder_returns_one_k_channel_tensor_per_order():
    decoder = MultiScalarDecoder(dim=8, num_targets=3)
    data = SimpleNamespace(
        h_1=torch.randn(5, 8),
        h_2=torch.randn(7, 8),
        h_3=torch.randn(11, 8),
    )

    outputs = decoder(data)
    assert [tuple(output.shape) for output in outputs] == [
        (5, 3),
        (7, 3),
        (11, 3),
    ]


def test_graph_multiscalar_adapter_pools_channels_without_vector_semantics():
    # Two graphs. Order-1 and order-2 contributions are summed independently
    # for each of the three scalar targets. normalize_by=None keeps exact sums.
    data = _IndexableNamespace(
        order_1_pred=torch.tensor(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
            ]
        ),
        order_2_pred=torch.tensor(
            [
                [4.0, 40.0, 400.0],
                [5.0, 50.0, 500.0],
            ]
        ),
        x_atm_batch=torch.tensor([0, 0, 1]),
        x_bnd_batch=torch.tensor([0, 1]),
    )

    model = GraphMultiScalarAdapter(
        _OrderContributionModel(),
        num_targets=3,
        normalize_by=None,
    )
    prediction = model(data)

    expected = torch.tensor(
        [
            [7.0, 70.0, 700.0],
            [8.0, 80.0, 800.0],
        ]
    )
    torch.testing.assert_close(prediction, expected)


def test_graph_multiscalar_validation_requires_b_by_k():
    task = GNNTask.graph_multiscalar(num_targets=3, target_key="y")
    task.validate_prediction_and_target(
        torch.zeros(4, 3),
        torch.ones(4, 3),
    )

    with pytest.raises(RuntimeError, match="prediction shape"):
        task.validate_prediction_and_target(
            torch.zeros(4, 2),
            torch.ones(4, 3),
        )


def test_default_decoder_builder_supports_multiscalar():
    decoder = build_default_decoder(
        "multiscalar",
        dim=8,
        out_dim=3,
    )
    assert isinstance(decoder, MultiScalarDecoder)
    assert decoder.num_targets == 3


def test_multiscalar_decoder_and_adapter_support_backpropagation():
    class _DecodedOrderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = MultiScalarDecoder(dim=8, num_targets=3)

        def forward(self, data):
            return self.decoder(data)

    data = _IndexableNamespace(
        h_1=torch.randn(5, 8),
        h_2=torch.randn(4, 8),
        h_3=torch.randn(6, 8),
        x_atm_batch=torch.tensor([0, 0, 1, 1, 1]),
        x_bnd_batch=torch.tensor([0, 0, 1, 1]),
        x_ang_batch=torch.tensor([0, 0, 0, 1, 1, 1]),
    )
    model = GraphMultiScalarAdapter(
        _DecodedOrderModel(),
        num_targets=3,
        normalize_by=None,
    )

    prediction = model(data)
    loss = torch.nn.functional.mse_loss(prediction, torch.randn(2, 3))
    loss.backward()

    assert prediction.shape == (2, 3)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_build_task_model_selects_multiscalar_decoder(monkeypatch):
    captured = {}

    def fake_build_model(**kwargs):
        captured.update(kwargs)
        return _DirectModel(num_targets=kwargs["out_dim"])

    monkeypatch.setattr("catalyst.ml.gnn.tasks.build_model", fake_build_model)

    task = GNNTask.graph_multiscalar(num_targets=3, target_key="y")
    model = build_task_model(
        task=task,
        preset="alignn",
        num_species=4,
        dim=16,
        num_convs=2,
    )

    assert isinstance(model, GraphMultiScalarAdapter)
    assert captured["decoder_type"] == "multiscalar"
    assert captured["output_type"] == "scalar"
    assert captured["output_level"] == "graph"
    assert captured["out_dim"] == 3
