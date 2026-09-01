"""Coverage matrix for all public Catalyst GNNTask families."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import catalyst.ml.gnn.tasks as tasks
from catalyst.ml.gnn import GNNTask, GraphMultiScalarAdapter, VectorChannelAdapter, build_task_model, task_from_parameters


@pytest.mark.parametrize(
    "task,expected",
    [
        (GNNTask.graph_scalar(), ("graph_scalar", "graph", "scalar", "exact", "scalar")),
        (GNNTask.graph_multiscalar(num_targets=3), ("graph_multiscalar", "graph", "scalar", "exact", "scalar")),
        (GNNTask.node_scalar(), ("node_scalar", "node", "scalar", "node", "scalar")),
        (GNNTask.node_vector(), ("node_vector", "node", "vector", "node", "vector")),
        (GNNTask.graph_vector(), ("graph_vector", "graph", "vector", "exact", "vector")),
        (GNNTask.scalar_gradient(), ("scalar_gradient", "node", "scalar_gradient", "node", "gradient")),
    ],
    ids=lambda item: item.name if isinstance(item, GNNTask) else None,
)
def test_every_task_applies_an_explicit_supervised_backend_contract(task, expected):
    name, level, output_type, accumulate, output_key = expected
    params = {"model_dict": {"prediction_params": {}}}
    task.apply_to_catalyst_parameters(params)

    assert task.name == name
    assert task.output_level == level
    assert task.output_type == output_type
    assert params["model_dict"]["accumulate_loss"] == accumulate
    assert params["model_dict"]["prediction_params"]["output_key"] == output_key
    assert params["model_dict"]["prediction_params"]["channel_mode"] == "target"


@pytest.mark.parametrize(
    "task",
    [
        GNNTask.graph_scalar(target_key="a"),
        GNNTask.graph_multiscalar(num_targets=3, target_key="b", target_names=("x", "y", "z")),
        GNNTask.node_scalar(target_key="c"),
        GNNTask.node_vector(target_key="d"),
        GNNTask.graph_vector(target_key="e"),
        GNNTask.scalar_gradient(target_key="f"),
    ],
    ids=lambda task: task.name,
)
def test_every_task_round_trips_through_parameters(task):
    params = {"model_dict": {"prediction_params": {}}}
    task.apply_to_catalyst_parameters(params)
    restored = task_from_parameters(params)
    assert restored == task


def test_valid_shape_contracts_for_all_task_families():
    GNNTask.graph_scalar().validate_prediction_and_target(
        torch.zeros(4), torch.zeros(4, 1)
    )
    GNNTask.graph_multiscalar(num_targets=3).validate_prediction_and_target(
        torch.zeros(4, 3), torch.zeros(12)
    )
    GNNTask.node_scalar().validate_prediction_and_target(
        torch.zeros(7), torch.zeros(7, 1)
    )
    GNNTask.node_vector().validate_prediction_and_target(
        torch.zeros(7, 3), torch.zeros(7, 3)
    )
    GNNTask.graph_vector().validate_prediction_and_target(
        torch.zeros(4, 3), torch.zeros(4, 3)
    )
    GNNTask.scalar_gradient().validate_prediction_and_target(
        torch.zeros(7, 3), torch.zeros(7, 3)
    )



def test_graph_vector_targets_batch_as_b_by_3():
    """Each graph must store a graph-vector target as [1, 3], not bare [3]."""
    graph_a = Data(
        x=torch.zeros((2, 1)),
        target_vector=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
    )
    graph_b = Data(
        x=torch.zeros((3, 1)),
        target_vector=torch.tensor([[-1.0, 0.5, 4.0]], dtype=torch.float32),
    )

    batch = next(iter(DataLoader([graph_a, graph_b], batch_size=2)))

    assert tuple(batch.target_vector.shape) == (2, 3)
    GNNTask.graph_vector(target_key="target_vector").validate_prediction_and_target(
        torch.zeros((2, 3), dtype=torch.float32),
        batch.target_vector,
    )

def test_vector_tasks_reject_multivector_channels_consistently():
    with pytest.raises(ValueError, match="vector_channels > 1"):
        GNNTask.node_vector(vector_channels=2)
    with pytest.raises(ValueError, match="vector_channels > 1"):
        GNNTask.graph_vector(vector_channels=2)


class _Identity(nn.Module):
    def forward(self, data):
        return data


@pytest.mark.parametrize(
    "task,model_type,expected_wrapper",
    [
        (GNNTask.node_vector(), "equivariant", VectorChannelAdapter),
        (GNNTask.graph_vector(), "equivariant", VectorChannelAdapter),
        (GNNTask.graph_multiscalar(num_targets=3), "generic", GraphMultiScalarAdapter),
    ],
    ids=["node_vector", "graph_vector", "graph_multiscalar"],
)
def test_task_model_builder_attaches_required_adapters(monkeypatch, task, model_type, expected_wrapper):
    monkeypatch.setattr(tasks, "build_model", lambda **kwargs: _Identity())
    model = build_task_model(
        task=task,
        model_type=model_type,
        num_species=2,
        cutoff=4.0,
        dim=16,
        num_convs=1,
    )
    assert isinstance(model, expected_wrapper)
    assert getattr(model, "_catalyst_task") == task


def test_scalar_gradient_builder_requires_equivariant_route(monkeypatch):
    monkeypatch.setattr(tasks, "build_model", lambda **kwargs: _Identity())
    with pytest.raises(ValueError, match="equivariant"):
        build_task_model(
            task=GNNTask.scalar_gradient(),
            model_type="generic",
            num_species=1,
            cutoff=4.0,
            dim=16,
            num_convs=1,
        )


def test_graph_multiscalar_aliases_resolve_to_canonical_task():
    canonical = GNNTask.graph_multiscalar(num_targets=3)
    assert GNNTask.graph_scalar_multichannel(num_targets=3) == canonical
    assert GNNTask.scalar_multichannel(num_targets=3) == canonical
    assert GNNTask.from_name("multiscalar", num_targets=3) == canonical
