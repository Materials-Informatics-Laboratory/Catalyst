"""Focused checkpoint round-trip equivalence test."""

from __future__ import annotations

import torch
from torch import nn

from catalyst.ml.gnn.GNN import GNN


def test_checkpoint_roundtrip_preserves_predictions_exactly(tmp_path):
    torch.manual_seed(31)
    model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 2))
    wrapper = GNN(model, torch.device("cpu"))
    x = torch.randn(6, 4)
    before = wrapper.model(x).detach().clone()

    parameters = {
        "device_dict": {"device": "cpu", "use_amp": False},
        "io_dict": {
            "model_dir": str(tmp_path),
            "remove_old_model": False,
            "checkpoint_keep_last": None,
        },
        "model_dict": {
            "model": wrapper,
            "optimizer_params": {
                "optimizer": "Adam",
                "implementation": "default",
                "params_group": {"lr": 1.0e-3},
            },
        },
    }
    wrapper.save_checkpoint(parameters, epoch=3)

    torch.manual_seed(99)
    restored = GNN(
        nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 2)),
        torch.device("cpu"),
    )
    restored.load_checkpoint(tmp_path / "checkpoint_epoch_3.pt", load_optimizer=False)
    after = restored.model(x).detach()

    assert torch.equal(before, after)
