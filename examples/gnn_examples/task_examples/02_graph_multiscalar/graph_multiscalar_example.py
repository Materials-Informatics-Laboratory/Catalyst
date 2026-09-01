"""Full Catalyst task example: graph_multiscalar on synthetic generic graphs.

The output channels are independent invariant scalar properties, not Cartesian
vector components.  Settings are read from ``config.json`` beside this file.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

TASK_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_EXAMPLES_DIR))
from _common import (  # noqa: E402
    infer_num_graphs,
    load_example_config,
    make_generic_graph,
    run_backend_task_example,
    scatter_mean,
)

from catalyst.ml.gnn import GNNTask, build_task_model


class GraphMultiScalarReadout(nn.Module):
    def __init__(self, dim: int, num_targets: int):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, num_targets))

    def forward(self, data):
        node_values = self.head(data.h_1)
        return scatter_mean(node_values, data.batch, infer_num_graphs(data))


def build_dataset(config: dict):
    dataset = config["dataset"]
    target_cfg = config["target"]
    n_types = int(dataset["n_types"])
    matrix = torch.as_tensor(target_cfg["composition_matrix"], dtype=torch.float32)
    offsets = torch.as_tensor(target_cfg["offsets"], dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[1] != n_types or matrix.shape[0] != offsets.numel():
        raise ValueError("Target matrix must have shape [num_targets, n_types].")

    graphs = []
    for idx in range(int(dataset["num_graphs"])):
        graph = make_generic_graph(
            seed=int(dataset["graph_seed_base"]) + idx,
            n_nodes=int(dataset["n_nodes_base"]) + (idx % int(dataset["n_nodes_cycle"])),
            n_types=n_types,
            k=int(dataset["knn_k"]),
        )
        composition = graph.node_G.float().mean(dim=0)
        targets = offsets + matrix @ composition
        graph.target_scalars = targets.float()
        graph.y = graph.target_scalars.clone()
        graph.gid = f"graph_multiscalar_{idx:03d}"
        graphs.append(graph)
    return graphs


def main() -> None:
    config = load_example_config(Path(__file__))
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    task_cfg = config["task"]
    target_names = tuple(str(value) for value in task_cfg["target_names"])
    num_targets = len(target_names)
    task = GNNTask.graph_multiscalar(
        num_targets=num_targets,
        target_key=str(task_cfg["target_key"]),
        target_names=target_names,
        normalize_by=task_cfg.get("normalize_by"),
    )

    model_cfg = config["model"]
    model = build_task_model(
        task=task,
        model_type="generic",
        encoder_type="generic",
        processor_type="order",
        conv_type=str(model_cfg["conv_type"]),
        decoder=GraphMultiScalarReadout(dim=int(model_cfg["dim"]), num_targets=num_targets),
        encode_3body=bool(model_cfg["encode_3body"]),
        num_species=int(config["dataset"]["n_types"]),
        cutoff=float(model_cfg["cutoff"]),
        dim=int(model_cfg["dim"]),
        num_convs=int(model_cfg["num_convs"]),
        aggr_scheme=str(model_cfg["aggr_scheme"]),
    )

    train_cfg = config["training"]
    run_backend_task_example(
        task=task,
        model=model,
        graphs=build_dataset(config),
        output_dir=Path(__file__).resolve().parent / str(config["output_dir"]),
        n_train=int(train_cfg["n_train"]),
        n_validation=int(train_cfg["n_validation"]),
        epochs=int(train_cfg["epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        batch_size=int(train_cfg["batch_size"]),
        plot_kind="graph_multiscalar",
        target_key=str(task_cfg["target_key"]),
        target_names=target_names,
    )
    print("graph_multiscalar task example completed.")


if __name__ == "__main__":
    main()
