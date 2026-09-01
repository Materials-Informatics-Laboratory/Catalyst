"""Smoke example 08: scalar_gradient task with live coordinate autograd."""

from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from torch_geometric.loader import DataLoader

from catalyst.graph.alignnd import alignn_gen
from catalyst.ml.gnn import GNNTask, build_task_model, validate_task_batch


def make_graph():
    vector = np.array([2.6, 0.4, 0.2], dtype=float)
    atoms = Atoms(
        symbols=["Al", "Cu"],
        positions=np.vstack((-0.5 * vector, 0.5 * vector)),
        cell=np.eye(3) * 12.0,
        pbc=False,
    )
    graph = alignn_gen(
        {
            "type": "alignnd",
            "raw_data": atoms,
            "element_list": ["Al", "Cu"],
            "neighbor_params": [5.0, -1],
            "include_angs": False,
            "is_dihedral": False,
            "store_raw_data": False,
            "use_pt": False,
            "include_equivariant_fields": True,
            "require_bonds": True,
            "require_angles": False,
        }
    )
    graph.target_vector = torch.zeros((2, 3), dtype=torch.float32)
    graph.y = graph.target_vector.clone()
    return graph


def main() -> None:
    torch.manual_seed(18)
    task = GNNTask.scalar_gradient(target_key="target_vector", output_key="gradient")
    model = build_task_model(
        task=task,
        model_type="equivariant",
        num_species=2,
        cutoff=5.0,
        dim=24,
        num_convs=1,
        return_dict=True,
        rbf_dim=16,
        gradient_sign="negative",
        decoder_kwargs={"create_graph": True, "retain_graph": True},
    )
    batch = next(iter(DataLoader([make_graph()], batch_size=1, follow_batch=["x_atm", "x_bnd"])))
    validate_task_batch(task=task, model=model, batch=batch)
    output = model(batch)
    assert isinstance(output, dict)
    assert tuple(output["gradient"].shape) == (2, 3)
    assert output["gradient"].requires_grad
    print("08_equivariant_scalar_gradient_smoke passed.")


if __name__ == "__main__":
    main()
