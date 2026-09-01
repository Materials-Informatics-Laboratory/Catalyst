"""Smoke example 06: generic node_scalar task with a real modular GNN."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch import nn
from torch_geometric.loader import DataLoader

from catalyst.graph.generic_build import generic_graph_gen
from catalyst.ml.gnn import GNNTask, build_task_model, validate_task_batch


class NodeScalarReadout(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, data):
        return self.linear(data.h_1)[:, 0]


def make_graph():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.1, 1.1, 0.0], [0.0, 0.2, 1.2]],
        dtype=np.float32,
    )
    labels = np.eye(3, dtype=np.float32)[[0, 1, 2, 1]]
    tree = KDTree(positions)
    distances, indices = tree.query(positions, k=4)
    graph = generic_graph_gen(
        {
            "type": "generic_pairwise",
            "raw_data": positions,
            "params": {"dist": distances, "ind": indices, "g_nodes": labels},
            "line_graph": True,
            "include_angs": True,
            "include_self_edges": False,
        }
    )
    graph.target_scalar = torch.tensor([-0.5, 0.25, 1.0, 0.25], dtype=torch.float32)
    graph.y = graph.target_scalar.clone()
    return graph


def main() -> None:
    torch.manual_seed(16)
    task = GNNTask.node_scalar(target_key="target_scalar")
    model = build_task_model(
        task=task,
        model_type="generic",
        encoder_type="generic",
        processor_type="order",
        conv_type="gine",
        decoder=NodeScalarReadout(dim=24),
        encode_3body=True,
        num_species=3,
        cutoff=4.0,
        dim=24,
        num_convs=1,
    )
    batch = next(iter(DataLoader([make_graph()], batch_size=1, follow_batch=["node_G", "node_A", "edge_A"])))
    validate_task_batch(task=task, model=model, batch=batch)
    pred = model(batch)
    assert tuple(pred.shape) == (4,)
    print("06_generic_node_scalar_smoke passed.")


if __name__ == "__main__":
    main()
