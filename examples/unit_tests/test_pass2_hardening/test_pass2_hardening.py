from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

import catalyst
from catalyst._version import __version__
from catalyst.ml.gnn import GNN, GNNTask
from catalyst.ml.gnn.modules.utils.data_manager import setup_dataloader
from catalyst.ml.training import run_training
from catalyst.ml.utils.distributed import _merge_result_dicts, validate_ddp_configuration
from catalyst.ml.utils.optimizer import set_optimizer
from catalyst.observer import Catalyst
from catalyst.properties.chemical_properties import (
    calc_reaction_enthalpy,
    check_num_elements,
    check_stoichiometry,
    get_structure_stoichiometry,
)
from catalyst.properties.structure_properties import unit_vector


def test_catalyst_version_and_system_info_have_one_source_of_truth():
    cat = Catalyst()
    info = cat.get_system_info()
    assert cat.version == __version__ == catalyst.__version__
    assert isinstance(info, dict)
    assert info["logical_count"] >= 1
    assert cat.parameters["device_dict"]["system_info"] == info


def test_vector_tasks_reject_unimplemented_multivector_contract():
    with pytest.raises(ValueError, match="vector_channels"):
        GNNTask.node_vector(vector_channels=2)
    with pytest.raises(ValueError, match="vector_channels"):
        GNNTask.graph_vector(vector_channels=3)


def test_ddp_configuration_fails_early_for_cpu_mode():
    params = {
        "device_dict": {
            "run_ddp": True,
            "world_size": 2,
            "device": "cpu",
            "ddp_backend": "gloo",
        }
    }
    with pytest.raises(ValueError, match="CUDA devices only"):
        validate_ddp_configuration(params, rank=0)


def test_gathered_ddp_result_dicts_are_flattened_consistently():
    merged = _merge_result_dicts(
        [
            {"gids": [1, 2], "pred": [[0.1], [0.2]], "vec": False},
            {"gids": [3], "pred": [[0.3]], "vec": False},
        ]
    )
    assert merged == {
        "gids": [1, 2, 3],
        "pred": [[0.1], [0.2], [0.3]],
        "vec": False,
    }


def test_optimizer_foreach_implementation_is_forwarded():
    model = nn.Linear(2, 1)
    params = {
        "model_dict": {
            "optimizer_params": {
                "optimizer": "AdamW",
                "implementation": "foreach",
                "params_group": {"params": model.parameters(), "lr": 1.0e-3},
            }
        }
    }
    optimizer = set_optimizer(params)
    assert optimizer.defaults.get("foreach") is True


def test_compile_model_uses_requested_backend_and_dynamic_shapes(monkeypatch):
    captured = {}

    def fake_compile(model, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    wrapper = GNN(nn.Linear(2, 1), "cpu")
    wrapper.compile_model(backend="inductor", mode="default", dynamic=True)
    assert captured["backend"] == "inductor"
    assert captured["dynamic"] is True


def _loader_cat(**loader_overrides):
    loader = {
        "num_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "prefetch_to_device": False,
        "batch_mode": "graphs",
        "max_nodes": None,
        "max_edges": None,
        "dynamic_batch_skip_too_big": False,
        "dynamic_batch_num_steps": None,
    }
    loader.update(loader_overrides)
    return SimpleNamespace(
        parameters={
            "device_dict": {
                "run_ddp": False,
                "pin_memory": False,
                "device": "cpu",
            },
            "loader_dict": loader,
            "sampling_dict": {"sampling_seed": 123},
        }
    )


def test_dynamic_node_budget_batching_respects_budget_for_normal_samples():
    data = [
        Data(x=torch.ones(n, 1), edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=n)
        for n in (2, 3, 4, 2)
    ]
    cat = _loader_cat(batch_mode="nodes", max_nodes=5)
    loader = setup_dataloader(
        data,
        cat,
        {"shuffle": False, "batch_size": 99, "epoch": 0},
    )
    sizes = [int(batch.num_nodes) for batch in loader]
    assert sizes == [5, 4, 2]
    assert all(size <= 5 for size in sizes)


def test_checkpoint_parameter_copy_skips_live_model_and_optimizer_params():
    model = nn.Linear(2, 1)
    params = {
        "device_dict": {"device": "cpu"},
        "model_dict": {
            "model": model,
            "optimizer_params": {
                "optimizer": "AdamW",
                "params_group": {"params": model.parameters(), "lr": 1.0e-3},
            },
        },
    }
    copied = GNN._checkpoint_parameter_copy(params)
    assert "model" not in copied["model_dict"]
    assert "params" not in copied["model_dict"]["optimizer_params"]["params_group"]
    assert copied["model_dict"]["optimizer_params"]["params_group"]["lr"] == 1.0e-3


class _IntervalTrainer:
    def __init__(self):
        self.model = nn.Linear(1, 1)
        self.device = "cpu"
        self.optimizer = None
        self.training_loader = SimpleNamespace(sampler=None)
        self.validation_loader = SimpleNamespace(sampler=None)
        self.train_calls = 0
        self.validation_calls = 0

    def load_data(self, *args, **kwargs):
        return None

    def set_optimizer_(self, parameters):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def set_dataloader(self, *args, **kwargs):
        return None

    def train(self, training_dict):
        self.train_calls += 1
        return 1.0 / self.train_calls

    def validate(self, parameters, rank=0):
        self.validation_calls += 1
        return 1.0 / self.validation_calls

    def save_checkpoint(self, *args, **kwargs):
        return None

    def _core_model(self):
        return self.model


def test_validation_interval_skips_unneeded_full_validation_passes(tmp_path: Path):
    samples = tmp_path / "samples"
    samples.mkdir()
    trainer = _IntervalTrainer()
    params = {
        "device_dict": {
            "device": "cpu",
            "run_ddp": False,
            "world_size": 1,
            "find_unused_parameters": False,
            "ddp_backend": "nccl",
        },
        "io_dict": {
            "main_path": str(tmp_path),
            "samples_dir": str(samples),
            "graph_read_format": -1,
            "training_info_nwrite_steps": 0,
        },
        "model_dict": {
            "model": trainer,
            "restart_training": False,
            "num_epochs": 5,
            "validation_interval": 2,
            "compile_model": False,
            "patience": 0,
            "worsen_tolerance": 0.05,
            "strict_loss_policy": False,
            "max_deltas": 0,
            "train_delta": 0.0,
            "train_tolerance": -1.0,
        },
    }
    run_training(0, SimpleNamespace(parameters=params))
    assert trainer.train_calls == 5
    # Epochs 2 and 4, plus the final epoch 5.
    assert trainer.validation_calls == 3


def test_chemical_helpers_validate_edge_cases_and_use_requested_delta():
    assert check_num_elements([["Fe", 0.5], ["Ni", 0.5]], [[ ["Fe", 0.5], ["Ni", 0.5] ]]) == 1
    assert check_num_elements([["Fe", 0.5], ["Ni", 0.5]], [[ ["Co", 0.5], ["Cu", 0.5] ]]) == 0

    main = [["A", 0.60], ["B", 0.40]]
    other = [[["A", 0.55], ["B", 0.45]]]
    assert check_stoichiometry(main, other, delta=0.11) == 1
    assert check_stoichiometry(main, other, delta=0.05) == 0

    with pytest.raises(ValueError, match="empty structure"):
        from ase import Atoms
        get_structure_stoichiometry(Atoms())

    assert calc_reaction_enthalpy([1.0, 0.2, 0.4], n_systems=3) == pytest.approx(0.7)
    with pytest.raises(ValueError, match="Expected 3 energies"):
        calc_reaction_enthalpy([1.0, 0.2], n_systems=3)


def test_unit_vector_rejects_zero_length_input():
    with pytest.raises(ValueError, match="zero-length"):
        unit_vector([0.0, 0.0, 0.0])
