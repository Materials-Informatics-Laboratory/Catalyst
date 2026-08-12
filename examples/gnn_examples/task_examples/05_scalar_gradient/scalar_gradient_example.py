"""Full Catalyst task example: scalar_gradient on a harmonic dimer dataset.

The model learns an invariant scalar potential while training against its
negative coordinate gradient.  Synthetic force labels come from an analytical
Al--Cu harmonic dimer potential. Settings are read from ``config.json``.
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
    potential = config["harmonic_potential"]
    count = int(dataset["num_graphs"])
    rng = np.random.default_rng(int(dataset["direction_seed"]))
    r0 = float(potential["equilibrium_distance"])
    spring = float(potential["spring_constant"])
    d_min = float(dataset["distance_min"])
    d_max = float(dataset["distance_max"])

    graphs = []
    for idx, direction in enumerate(random_unit_vectors(rng, count)):
        fraction = idx / max(count - 1, 1)
        distance = d_min + (d_max - d_min) * fraction
        vector = direction * distance
        graph = make_dimer_graph(vector=vector, gid=f"scalar_gradient_{idx:03d}")

        magnitude = spring * (distance - r0)
        force_al = magnitude * direction
        force_cu = -force_al
        forces = np.vstack((force_al, force_cu))
        graph.target_vector = torch.as_tensor(forces, dtype=torch.float32)
        graph.y = graph.target_vector.clone()
        graph.reference_energy = torch.tensor(
            [0.5 * spring * (distance - r0) ** 2], dtype=torch.float32
        )
        graphs.append(graph)
    return graphs


def main() -> None:
    config = load_example_config(Path(__file__))
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    task_cfg = config["task"]
    task = GNNTask.scalar_gradient(
        target_key=str(task_cfg["target_key"]),
        output_key=str(task_cfg["output_key"]),
    )
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
        gradient_sign=str(model_cfg["gradient_sign"]),
        decoder_kwargs={
            "reduce": str(model_cfg["reduce"]),
            "create_graph": bool(model_cfg["create_graph"]),
            "retain_graph": bool(model_cfg["retain_graph"]),
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
        plot_kind="scalar_gradient",
        target_key=str(task_cfg["target_key"]),
    )
    print("scalar_gradient task example completed.")


if __name__ == "__main__":
    main()
