"""
Smoke example 01: generic graph_scalar task.

This is a minimal developer smoke test for the public task API. It validates:
    - GNNTask.graph_scalar(...)
    - Catalyst(task=...) staged task/configuration validation
    - validate_task_batch(...)

It intentionally uses a tiny dummy model so it can run even when the full graph
builder dependencies are unavailable.

Run:
    python examples/gnn_examples/smoke/01_generic_graph_scalar_smoke.py
"""

from __future__ import annotations

import torch
from torch import nn

from catalyst.observer import Catalyst

from catalyst.ml.gnn.tasks import GNNTask, validate_task_batch


class TinyGraphScalarModel(nn.Module):
    def forward(self, batch):
        x = batch.x.float()
        batch_index = getattr(batch, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        n_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 1
        out = x.new_zeros((n_graphs, 1))
        out.index_add_(0, batch_index, x[:, :1])
        counts = x.new_zeros((n_graphs, 1))
        counts.index_add_(0, batch_index, x.new_ones((x.size(0), 1)))
        return (out / counts.clamp_min(1.0)).view(-1)


class TinyBatch:
    def __init__(self):
        self.x = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)
        self.batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        self.target_scalar = torch.tensor([0.5, 2.5], dtype=torch.float32)

    def to(self, device):
        self.x = self.x.to(device)
        self.batch = self.batch.to(device)
        self.target_scalar = self.target_scalar.to(device)
        return self


def main() -> None:
    task = GNNTask.graph_scalar(target_key="target_scalar")

    cat = Catalyst(task=task)
    parameters = cat.parameters

    assert parameters["model_dict"]["task"] == "graph_scalar"
    assert parameters["model_dict"]["accumulate_loss"] == "exact"
    assert parameters["model_dict"]["prediction_params"]["target_key"] == "target_scalar"

    validate_task_batch(
        task=task,
        model=TinyGraphScalarModel(),
        batch=TinyBatch(),
        print_summary=True,
    )

    print("01_generic_graph_scalar_smoke passed.")


if __name__ == "__main__":
    main()
