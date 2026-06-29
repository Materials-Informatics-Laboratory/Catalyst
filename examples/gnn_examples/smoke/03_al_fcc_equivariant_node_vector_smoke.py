"""
Smoke example 03: Al FCC equivariant node_vector task.

This smoke test exercises the task-native route:

    build_task_model(
        task=GNNTask.node_vector(...),
        model_type="equivariant",
        ...
    )

The task supplies output_type="vector", output_level="node", out_dim=1 and wraps
the raw [N, 1, 3] equivariant output as [N, 3].

Run:
    python examples/gnn_examples/smoke/03_al_fcc_equivariant_node_vector_smoke.py
"""

from __future__ import annotations

import torch
from torch import nn

from catalyst.ml.gnn.tasks import GNNTask, build_task_model, validate_task_batch


def make_al_force_graph():
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from catalyst.graph.alignnd import alignn_gen

    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((1, 1, 1))
    atoms.pbc = True
    atoms.calc = EMT()
    forces = torch.as_tensor(atoms.get_forces(), dtype=torch.float32).reshape(-1, 3)

    graph = alignn_gen(
        {
            "type": "alignnd",
            "raw_data": atoms,
            "node_labels": None,
            "element_list": ["Al"],
            "neighbor_params": [3.5, 12],
            "is_dihedral": False,
            "store_raw_data": False,
            "use_pt": False,
            "include_angs": True,
            "cpu_cores": 1,
            "store_atoms_type": "ase-atoms",
            "include_equivariant_fields": True,
            "include_edge_geometry": True,
            "auto_retry_graph": True,
            "max_graph_attempts": 4,
            "require_bonds": True,
            "require_angles": False,
            "require_dihedrals": False,
        }
    )
    if isinstance(graph, (list, tuple)):
        graph = graph[0]

    if getattr(graph, "edge_index", None) is None and getattr(graph, "edge_index_G", None) is not None:
        graph.edge_index = graph.edge_index_G

    if getattr(graph, "shifts", None) is None and getattr(graph, "edge_index", None) is not None:
        graph.shifts = torch.zeros((graph.edge_index.size(1), 3), dtype=torch.long)

    if getattr(graph, "pbc", None) is None:
        graph.pbc = torch.tensor([True, True, True], dtype=torch.bool)

    if getattr(graph, "num_nodes", None) is None:
        graph.num_nodes = int(forces.size(0))

    graph.gid = "al_fcc_smoke_node_vector"
    graph.forces_eVA = forces.clone()
    graph.target_vector = forces.clone()
    graph.y = forces.clone()
    return graph


def main() -> None:
    from torch_geometric.loader import DataLoader

    task = GNNTask.node_vector(
        target_key="target_vector",
        vector_channels=1,
        squeeze_single_vector_channel=True,
    )

    parameters = {"model_dict": {"prediction_params": {}}}
    task.apply_to_catalyst_parameters(parameters)

    model = build_task_model(
        task=task,
        model_type="equivariant",
        return_dict=False,
        num_species=1,
        cutoff=3.5,
        dim=32,
        num_convs=1,
        act=nn.SiLU(),
    )

    graph = make_al_force_graph()
    batch = next(iter(DataLoader([graph], batch_size=1)))
    validate_task_batch(task=task, model=model, batch=batch, print_summary=True)

    print("03_al_fcc_equivariant_node_vector_smoke passed.")


if __name__ == "__main__":
    main()
