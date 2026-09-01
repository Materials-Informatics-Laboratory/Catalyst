from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from catalyst.observer import Catalyst, CatalystParameterError
from catalyst.ml.gnn import GNNTask, build_task_model
import catalyst.ml.gnn.tasks as task_module


def _runtime_overrides(tmp_path: Path):
    return {
        "io_dict": {
            "main_path": str(tmp_path),
            "data_dir": str(tmp_path),
            "samples_dir": str(tmp_path),
            "model_dir": str(tmp_path),
            "results_dir": str(tmp_path),
        },
        "model_dict": {
            "loss_params": {"function": "MSELoss"},
            "optimizer_params": {
                "optimizer": "Adam",
                "params_group": {"lr": 1.0e-3},
            },
        },
    }


def test_constructor_merges_json_then_python_overrides_and_resolves_paths(tmp_path):
    config = {
        "workflow": {"train": True},
        "catalyst_parameters": {
            "io_dict": {"main_path": ".", "data_dir": "graphs"},
            "loader_dict": {"batch_size": [2, 2]},
            "model_dict": {"num_epochs": 5, "loss_params": {"function": "MSELoss"}},
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    cat = Catalyst(
        parameter_file=path,
        parameters={"model_dict": {"num_epochs": 9}},
    )

    assert cat.parameters["model_dict"]["num_epochs"] == 9
    assert Path(cat.parameters["io_dict"]["main_path"]) == tmp_path
    assert Path(cat.parameters["io_dict"]["data_dir"]) == tmp_path / "graphs"
    assert cat.validation_status()["stage"] == "config"


def test_unknown_parameter_is_rejected_with_suggestion():
    with pytest.raises(CatalystParameterError, match="batch_size"):
        Catalyst(parameters={"loader_dict": {"batch_szie": [4, 4]}})


def test_invalid_set_params_is_atomic():
    cat = Catalyst(parameters={"loader_dict": {"batch_size": [4, 4]}})
    before = list(cat.parameters["loader_dict"]["batch_size"])

    with pytest.raises(CatalystParameterError):
        cat.set_params({"loader_dict": {"batch_size": [0, 4]}}, save_params=False)

    assert cat.parameters["loader_dict"]["batch_size"] == before


def test_task_can_be_bound_after_constructor():
    cat = Catalyst(parameters={"model_dict": {"loss_params": {"function": "MSELoss"}}})
    assert cat.task is None

    task = GNNTask.node_vector(target_key="target_vector")
    cat.set_task(task)

    assert cat.task == task
    assert cat.parameters["model_dict"]["task"] == "node_vector"
    assert cat.parameters["model_dict"]["accumulate_loss"] == "node"
    assert cat.parameters["model_dict"]["prediction_params"]["target_key"] == "target_vector"
    assert cat.validation_status()["stage"] == "task"


def test_explicit_task_conflict_is_rejected():
    with pytest.raises(CatalystParameterError, match="accumulate_loss"):
        Catalyst(
            parameters={"model_dict": {"accumulate_loss": "node"}},
            task=GNNTask.graph_scalar(),
        )


def test_task_controlled_update_cannot_be_changed_after_binding():
    cat = Catalyst(task=GNNTask.graph_scalar(target_key="energy"))
    with pytest.raises(CatalystParameterError, match="target_key"):
        cat.set_params(
            {"model_dict": {"prediction_params": {"target_key": "forces"}}},
            save_params=False,
        )
    assert cat.parameters["model_dict"]["prediction_params"]["target_key"] == "energy"


def test_build_task_model_rejects_task_controlled_model_kwarg_conflict():
    task = GNNTask.graph_multiscalar(num_targets=3)
    with pytest.raises(ValueError, match="out_dim"):
        build_task_model(
            task=task,
            preset="alignn",
            out_dim=2,
            num_species=1,
            cutoff=3.5,
            dim=8,
            num_convs=1,
        )


def test_build_task_model_attaches_task_metadata(monkeypatch):
    task = GNNTask.graph_scalar()
    dummy = nn.Linear(2, 1)
    monkeypatch.setattr(task_module, "build_model", lambda **kwargs: dummy)

    model = build_task_model(
        task=task,
        model_type="generic",
        num_species=1,
        cutoff=3.5,
        dim=8,
        num_convs=1,
    )

    assert model._catalyst_task == task
    assert model._catalyst_model_kwargs["output_type"] == "scalar"
    assert model._catalyst_model_kwargs["output_level"] == "graph"


def test_final_training_preflight_can_run_after_late_task_binding(tmp_path):
    cat = Catalyst(parameters=_runtime_overrides(tmp_path))
    task = GNNTask.graph_scalar(target_key="target_scalar")
    cat.set_task(task)

    model = nn.Linear(2, 1)
    status = cat.validate_ready_for_training(model=model)

    assert status["stage"] == "ready"
    assert status["task"] == "graph_scalar"


def test_effective_parameters_can_be_saved_as_json(tmp_path):
    cat = Catalyst(parameters={"model_dict": {"loss_params": {"function": "MSELoss"}}})
    out = tmp_path / "effective_parameters.json"
    cat.save_parameters(out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["model_dict"]["model"] is None
    assert payload["model_dict"]["loss_params"]["function"] == "MSELoss"


def _is_cat_parameters_root(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "parameters"
        and isinstance(node.value, ast.Name)
        and node.value.id == "cat"
    )


def _contains_cat_parameter_target(node):
    current = node
    while isinstance(current, ast.Subscript):
        current = current.value
    return _is_cat_parameters_root(current)


def test_public_workflow_examples_do_not_mutate_cat_parameters_directly():
    # Anchor to the examples tree rather than assuming a repository checkout.
    # This supports both:
    #   repo/examples/unit_tests/...
    # and a standalone copied examples tree:
    #   catalyst_test/unit_tests/...
    examples_root = Path(__file__).resolve().parents[2]
    example_files = [
        examples_root / "gnn_examples/alignn_examples/energy/al_fcc_alignn_energy_example.py",
        examples_root / "gnn_examples/alignn_examples/force/al_fcc_equivariant_force_catalyst_backend.py",
        examples_root / "gnn_examples/pre_training_workflow/example_workflow.py",
        *sorted((examples_root / "gnn_examples/task_examples").glob("[0-9][0-9]_*.py")),
    ]

    violations = []
    for path in example_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(_contains_cat_parameter_target(target) for target in targets):
                    violations.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update":
                if _contains_cat_parameter_target(node.func.value):
                    violations.append(f"{path.name}:{node.lineno}")

    assert violations == []


def test_graph_scalar_rejects_explicit_latent_channel_mode():
    with pytest.raises(Exception, match="channel_mode"):
        Catalyst(
            parameters={
                "model_dict": {
                    "prediction_params": {"channel_mode": "latent"}
                }
            },
            task=GNNTask.graph_scalar(target_key="y"),
        )
