"""Full Catalyst task example: graph_vector with an equivariant dimer model.

Each Al--Cu graph is assigned the displacement vector from Al to Cu. Rotating
the structure rotates the target, so this exercises true graph-vector semantics.
Settings are read from ``config.json`` beside this file.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

TASK_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_EXAMPLES_DIR))
from _common import (  # noqa: E402
    load_example_config,
    make_dimer_graph,
    random_unit_vectors,
    run_backend_task_example,
)

from catalyst.ml.gnn import GNNTask, build_task_model


def build_dataset(config: dict):
    dataset = config["dataset"]
    count = int(dataset["num_graphs"])
    rng = np.random.default_rng(int(dataset["direction_seed"]))
    d_min = float(dataset["distance_min"])
    d_max = float(dataset["distance_max"])

    graphs = []
    for idx, direction in enumerate(random_unit_vectors(rng, count)):
        fraction = idx / max(count - 1, 1)
        distance = d_min + (d_max - d_min) * fraction
        vector = direction * distance
        graph = make_dimer_graph(vector=vector, gid=f"graph_vector_{idx:03d}")
        graph.target_vector = torch.as_tensor(vector, dtype=torch.float32).reshape(1, 3)
        graph.y = graph.target_vector.clone()
        graphs.append(graph)
    return graphs


def main() -> None:
    config = load_example_config(Path(__file__))
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    task = GNNTask.graph_vector(target_key=str(config["task"]["target_key"]))
    model_cfg = config["model"]
    model = build_task_model(
        task=task,
        model_type="equivariant",
        num_species=int(model_cfg["num_species"]),
        cutoff=float(model_cfg["cutoff"]),
        dim=int(model_cfg["dim"]),
        num_convs=int(model_cfg["num_convs"]),
        return_dict=bool(model_cfg["return_dict"]),
        rbf_dim=int(model_cfg["rbf_dim"]),
        decoder_kwargs={
            "reduce": str(model_cfg["reduce"]),
            "vector_reduce": str(model_cfg["vector_reduce"]),
            "squeeze_vector_channels": bool(model_cfg["squeeze_vector_channels"]),
        },
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
        plot_kind="graph_vector",
        target_key=str(config["task"]["target_key"]),
    )
    print("graph_vector task example completed.")


if __name__ == "__main__":
    main()
