from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data

from catalyst.data.utils import safe_torch_load
from catalyst.graph.generic_build import generic_graph_gen
from catalyst.ml.gnn.modules.models.alignn import ALIGNN
from catalyst.ml.gnn.modules.utils.data_manager import _shared_sampler_seed
from catalyst.ml.training import run_active_learning, run_training
from catalyst.ml.utils.loss import MaxNpercent
from catalyst.properties.structure_properties import get_3body_angle


def test_generic_three_body_angles_are_returned_in_radians():
    points = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    edge_index_g = np.asarray([[0, 2], [1, 1]], dtype=np.int64)
    edge_index_a = np.asarray([[0], [1]], dtype=np.int64)
    angle = get_3body_angle(points, edge_index_g, edge_index_a)
    np.testing.assert_allclose(angle, [np.pi / 2.0], rtol=1e-6, atol=1e-6)


def test_generic_nbody_is_not_advertised_as_supported():
    with pytest.raises(ValueError, match="generic_pairwise"):
        generic_graph_gen({"type": "generic_nbody", "raw_data": {}})


def test_safe_torch_load_can_restore_custom_pyg_objects(tmp_path: Path):
    path = tmp_path / "graph.pt"
    original = Data(x=torch.tensor([[1.0], [2.0]]), gid="example")
    torch.save(original, path)
    loaded = safe_torch_load(path, map_location="cpu")
    assert isinstance(loaded, Data)
    assert loaded.gid == "example"
    torch.testing.assert_close(loaded.x, original.x)


def test_ddp_sampler_seed_comes_from_shared_sampling_seed():
    params = {"sampling_dict": {"sampling_seed": 112358}}
    assert _shared_sampler_seed(params) == 112358
    assert _shared_sampler_seed(params) == 112358


def test_shared_ddp_seed_produces_disjoint_rank_partitions():
    dataset = list(range(10))
    seed = _shared_sampler_seed({"sampling_dict": {"sampling_seed": 112358}})
    rank0 = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True, seed=seed)
    rank1 = DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=True, seed=seed)
    rank0.set_epoch(3)
    rank1.set_epoch(3)
    part0 = set(iter(rank0))
    part1 = set(iter(rank1))
    assert part0.isdisjoint(part1)
    assert part0 | part1 == set(range(10))


def test_max_n_percent_selects_largest_absolute_sample_errors():
    loss = MaxNpercent(percent=0.5, sub_function=nn.MSELoss())
    prediction = torch.tensor([0.0, 1.0, 10.0, 3.0])
    target = torch.zeros_like(prediction)
    # Worst 50% are errors 10 and 3: (100 + 9) / 2 = 54.5.
    torch.testing.assert_close(loss(prediction, target), torch.tensor(54.5))


def test_max_n_percent_ranks_multichannel_outputs_per_sample():
    loss = MaxNpercent(percent=0.5, sub_function=nn.L1Loss())
    prediction = torch.tensor([[1.0, 1.0], [10.0, 0.0], [2.0, 2.0], [0.0, 0.0]])
    target = torch.zeros_like(prediction)
    # Per-sample MAE scores are 1, 5, 2, 0, so rows 1 and 2 are selected.
    expected = torch.tensor((10.0 + 0.0 + 2.0 + 2.0) / 4.0)
    torch.testing.assert_close(loss(prediction, target), expected)


def test_legacy_alignn_class_import_resolves_to_current_builder():
    assert issubclass(ALIGNN, nn.Module)


def test_legacy_active_learning_fails_explicitly_instead_of_partial_execution():
    with pytest.raises(NotImplementedError, match="not part of the supported"):
        run_active_learning(0, None)


class _FakeTrainer:
    def __init__(self):
        self.model = nn.Linear(1, 1)
        self.device = "cpu"
        self.optimizer = None
        self.loaded_checkpoint = None
        self.train_calls = 0
        self.validation_calls = 0
        self.saved_epochs = []
        self.training_loader = SimpleNamespace(sampler=None)
        self.validation_loader = SimpleNamespace(sampler=None)

    def load_data(self, *args, **kwargs):
        return None

    def set_optimizer_(self, parameters):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def set_dataloader(self, cat, epoch=-1, **kwargs):
        return None

    def load_checkpoint(self, fname, map_location=None, load_optimizer=True, strict=True):
        self.loaded_checkpoint = str(fname)
        return 2

    def train(self, training_dict):
        self.train_calls += 1
        return 0.5

    def validate(self, parameters, rank=0):
        self.validation_calls += 1
        return 0.4

    def save_checkpoint(self, parameters, epoch, rank=0, fname=None):
        self.saved_epochs.append(int(epoch))

    def _core_model(self):
        return self.model


def test_restart_training_preserves_and_loads_checkpoint(tmp_path: Path):
    model_dir = tmp_path / "models" / "training"
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / "checkpoint_epoch_2.pt"
    checkpoint.write_bytes(b"checkpoint placeholder")

    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()

    trainer = _FakeTrainer()
    parameters = {
        "device_dict": {
            "device": "cpu",
            "run_ddp": False,
            "world_size": 1,
            "find_unused_parameters": False,
            "ddp_backend": "gloo",
        },
        "io_dict": {
            "main_path": str(tmp_path),
            "samples_dir": str(samples_dir),
            "graph_read_format": -1,
            "loaded_model_name": str(checkpoint),
            "training_info_nwrite_steps": 1,
            "model_dir": str(model_dir),
        },
        "model_dict": {
            "model": trainer,
            "restart_training": True,
            "num_epochs": 4,
            "patience": 0,
            "worsen_tolerance": 0.05,
            "strict_loss_policy": False,
            "max_deltas": 4,
            "train_delta": 0.0,
            "train_tolerance": -1.0,
        },
    }
    cat = SimpleNamespace(parameters=parameters)

    run_training(0, cat)

    assert checkpoint.exists(), "restart setup must not delete the checkpoint directory"
    assert trainer.loaded_checkpoint == str(checkpoint.resolve())
    assert trainer.train_calls == 1
    assert trainer.validation_calls == 1
    assert trainer.saved_epochs == [3]


class _RollbackTrainer(_FakeTrainer):
    def __init__(self):
        super().__init__()
        self._train_losses = iter([1.0, 2.0])
        self._valid_losses = iter([1.0, 2.0])
        with torch.no_grad():
            self.model.weight.fill_(0.0)
            self.model.bias.fill_(0.0)

    def train(self, training_dict):
        self.train_calls += 1
        with torch.no_grad():
            self.model.weight.fill_(float(self.train_calls))
        return next(self._train_losses)

    def validate(self, parameters, rank=0):
        self.validation_calls += 1
        return next(self._valid_losses)


def test_training_rolls_back_when_validation_worsens_beyond_tolerance(tmp_path: Path):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    trainer = _RollbackTrainer()
    parameters = {
        "device_dict": {
            "device": "cpu",
            "run_ddp": False,
            "world_size": 1,
            "find_unused_parameters": False,
            "ddp_backend": "gloo",
        },
        "io_dict": {
            "main_path": str(tmp_path),
            "samples_dir": str(samples_dir),
            "graph_read_format": -1,
            "training_info_nwrite_steps": 1,
        },
        "model_dict": {
            "model": trainer,
            "restart_training": False,
            "num_epochs": 2,
            "patience": 0,
            "worsen_tolerance": 0.05,
            "strict_loss_policy": False,
            "max_deltas": 4,
            "train_delta": 0.0,
            "train_tolerance": -1.0,
        },
    }
    cat = SimpleNamespace(parameters=parameters)

    run_training(0, cat)

    assert trainer.train_calls == 2
    assert trainer.validation_calls == 2
    assert trainer.saved_epochs == [0]
    torch.testing.assert_close(
        trainer.model.weight.detach(),
        torch.ones_like(trainer.model.weight),
    )
