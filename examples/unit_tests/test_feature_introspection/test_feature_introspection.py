from __future__ import annotations

import torch
from torch import nn

from catalyst.ml.gnn.modules.models import GNNBuilder, unwrap_feature_model
from catalyst.ml.gnn.tasks import VectorChannelAdapter
from catalyst.ml.gnn.modules.utils.predict import (
    accumulate_predictions,
    accumulate_predictions_from_parameters,
    prediction_kwargs_from_parameters,
)


class DummyData:
    pass


class SimpleEncoder(nn.Module):
    def forward(self, data):
        data.h_1 = data.x.clone()
        return data


class SimpleProcessor(nn.Module):
    def forward(self, data):
        data.h_1 = 2.0 * data.h_1
        return data


class SumDecoder(nn.Module):
    output_type = "scalar"

    def forward(self, data):
        return data.h_1.sum(dim=-1)


class VectorDecoder(nn.Module):
    output_type = "vector"

    def forward(self, data):
        # One explicit vector channel to exercise VectorChannelAdapter.
        return data.h_1[:, :3].unsqueeze(1)


class Wrapper:
    def __init__(self, model):
        self.model = model


def _data():
    data = DummyData()
    data.x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
        ]
    )
    return data


def test_forward_features_decode_features_matches_normal_forward():
    model = GNNBuilder(SimpleEncoder(), SimpleProcessor(), SumDecoder())

    direct = model(_data())
    features = model.forward_features(_data())
    split = model.decode_features(features)

    assert torch.equal(direct, split)
    assert torch.equal(
        features.h_1,
        2.0 * torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
            ]
        ),
    )


def test_latent_intervention_changes_only_decode_path():
    model = GNNBuilder(SimpleEncoder(), SimpleProcessor(), SumDecoder())
    data = _data()

    features = model.forward_features(data)
    baseline = model.decode_features(features).clone()

    features.h_1[:, 0] = 0.0
    intervened = model.decode_features(features)

    assert not torch.equal(baseline, intervened)
    assert torch.all(intervened < baseline)


def test_vector_task_adapter_preserves_feature_api_and_output_semantics():
    core = GNNBuilder(SimpleEncoder(), SimpleProcessor(), VectorDecoder())
    model = VectorChannelAdapter(core)

    direct = model(_data())
    features = model.forward_features(_data())
    split = model.decode_features(features)

    assert tuple(direct.shape) == (2, 3)
    assert torch.equal(direct, split)


def test_unwrap_feature_model_finds_feature_capable_model_through_wrapper():
    core = GNNBuilder(SimpleEncoder(), SimpleProcessor(), SumDecoder())
    wrapped = Wrapper(core)

    assert unwrap_feature_model(wrapped) is core


def test_unwrap_feature_model_prefers_semantic_task_wrapper():
    core = GNNBuilder(SimpleEncoder(), SimpleProcessor(), VectorDecoder())
    task_wrapper = VectorChannelAdapter(core)
    outer = Wrapper(task_wrapper)

    # The task wrapper implements the feature API and preserves its output
    # semantics, so it is the correct model for external intervention tooling.
    assert unwrap_feature_model(outer) is task_wrapper


def test_prediction_parameter_helper_matches_direct_accumulator():
    data = DummyData()
    data.target_scalar = torch.tensor([1.5, 2.5])
    pred = torch.tensor([1.25, 2.75])
    parameters = {
        "model_dict": {
            "accumulate_loss": "node",
            "loss_params": {"target_key": "legacy_target"},
            "prediction_params": {
                "target_key": "target_scalar",
                "channel_mode": None,
            },
        }
    }

    kwargs = prediction_kwargs_from_parameters(parameters)
    assert kwargs["target_key"] == "target_scalar"
    assert "channel_mode" not in kwargs

    expected = accumulate_predictions(
        pred,
        data,
        "node",
        return_y=True,
        target_key="target_scalar",
    )
    actual = accumulate_predictions_from_parameters(
        pred,
        data,
        parameters,
        return_y=True,
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[2] == expected[2]
