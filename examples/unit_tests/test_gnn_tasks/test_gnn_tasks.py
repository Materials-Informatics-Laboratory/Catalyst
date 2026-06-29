from __future__ import annotations

import torch
from torch import nn

import catalyst.ml.gnn.tasks as tasks
from catalyst.ml.gnn.tasks import GNNTask, VectorChannelAdapter, build_task_model, task_from_parameters


class ConstantVectorChannelModel(nn.Module):
    def forward(self, data):
        return torch.zeros(5, 1, 3)


class ConstantDictVectorChannelModel(nn.Module):
    def forward(self, data):
        return {"vector": torch.zeros(5, 1, 3)}


class IdentityModel(nn.Module):
    def forward(self, data):
        return data


def test_graph_scalar_applies_backend_contract():
    task = GNNTask.graph_scalar(target_key="target_scalar")
    parameters = {"model_dict": {"prediction_params": {}}}

    task.apply_to_catalyst_parameters(parameters)

    assert parameters["model_dict"]["task"] == "graph_scalar"
    assert parameters["model_dict"]["accumulate_loss"] == "exact"
    assert parameters["model_dict"]["prediction_params"]["target_key"] == "target_scalar"
    assert parameters["model_dict"]["prediction_params"]["output_key"] == "scalar"
    assert parameters["model_dict"]["prediction_params"]["prefer_equivariant_key"] == "scalar"


def test_node_vector_applies_backend_contract():
    task = GNNTask.node_vector(target_key="target_vector")
    parameters = {"model_dict": {"prediction_params": {}}}

    task.apply_to_catalyst_parameters(parameters)

    assert parameters["model_dict"]["task"] == "node_vector"
    assert parameters["model_dict"]["accumulate_loss"] == "node"
    assert parameters["model_dict"]["prediction_params"]["target_key"] == "target_vector"
    assert parameters["model_dict"]["prediction_params"]["output_key"] == "vector"
    assert parameters["model_dict"]["prediction_params"]["prefer_equivariant_key"] == "vector"


def test_vector_adapter_squeezes_single_channel_tensor():
    model = VectorChannelAdapter(ConstantVectorChannelModel())
    out = model(object())
    assert tuple(out.shape) == (5, 3)


def test_vector_adapter_squeezes_single_channel_dict():
    model = VectorChannelAdapter(ConstantDictVectorChannelModel())
    out = model(object())
    assert tuple(out.shape) == (5, 3)


def test_vector_adapter_rejects_three_vector_channels():
    class Bad(nn.Module):
        def forward(self, data):
            return torch.zeros(5, 3, 3)

    model = VectorChannelAdapter(Bad(), vector_channels=1)
    try:
        model(object())
    except RuntimeError as exc:
        assert "Unexpected number of vector channels" in str(exc)
    else:
        raise AssertionError("Expected VectorChannelAdapter to reject [N, 3, 3].")


def test_task_from_parameters_node_vector():
    parameters = {
        "model_dict": {
            "accumulate_loss": "node",
            "prediction_params": {
                "target_key": "target_vector",
                "output_key": "vector",
            },
        }
    }

    task = task_from_parameters(parameters)
    assert task.name == "node_vector"
    assert task.target_key == "target_vector"


def test_build_task_model_consumes_apply_task_model_kwargs(monkeypatch):
    captured = {}

    def fake_build_model(**kwargs):
        captured.update(kwargs)
        return IdentityModel()

    monkeypatch.setattr(tasks, "build_model", fake_build_model)

    task = GNNTask.graph_scalar()
    model = build_task_model(
        task=task,
        model_type="gnn_builder",
        apply_task_model_kwargs=False,
        num_species=1,
        cutoff=3.0,
        dim=16,
        num_convs=1,
    )

    assert isinstance(model, IdentityModel)
    assert "apply_task_model_kwargs" not in captured
    assert "output_type" not in captured
    assert "output_level" not in captured
    assert "out_dim" not in captured


def test_build_task_model_injects_task_kwargs_when_enabled(monkeypatch):
    captured = {}

    def fake_build_model(**kwargs):
        captured.update(kwargs)
        return IdentityModel()

    monkeypatch.setattr(tasks, "build_model", fake_build_model)

    task = GNNTask.node_vector(vector_channels=1)
    model = build_task_model(
        task=task,
        model_type="equivariant",
        return_dict=False,
        num_species=1,
        cutoff=3.0,
        dim=16,
        num_convs=1,
    )

    # node_vector wraps with VectorChannelAdapter.
    assert isinstance(model, VectorChannelAdapter)
    assert captured["model_type"] == "equivariant"
    assert captured["output_type"] == "vector"
    assert captured["output_level"] == "node"
    assert captured["out_dim"] == 1


def test_build_task_model_routes_preset_without_leaking_model_type(monkeypatch):
    captured = {}

    def fake_build_model(**kwargs):
        captured.update(kwargs)
        return IdentityModel()

    monkeypatch.setattr(tasks, "build_model", fake_build_model)

    task = GNNTask.graph_scalar()
    build_task_model(
        task=task,
        model_type="gnn_builder",
        preset="alignn",
        apply_task_model_kwargs=False,
        num_species=1,
        cutoff=3.0,
        dim=16,
        num_convs=1,
    )

    assert captured["preset"] == "alignn"
    assert "model_type" not in captured
    assert "apply_task_model_kwargs" not in captured
