"""
Smoke example 04: Al FCC ALIGNN graph_multiscalar task.

This smoke test exercises the canonical independent multi-scalar route:

    task = GNNTask.graph_multiscalar(num_targets=3, ...)
    model = build_task_model(task=task, preset="alignn", ...)

The three channels are independent invariant graph-level scalar targets. They
are not interpreted as a Cartesian vector.

Run:
    python examples/gnn_examples/smoke/04_al_fcc_alignn_graph_multiscalar_smoke.py
"""

from __future__ import annotations

import torch
from torch import nn

from catalyst.ml.gnn import GNNTask, GraphMultiScalarAdapter, build_task_model, validate_task_batch
from catalyst.ml.gnn.modules.decoders import MultiScalarDecoder


def make_al_graph(lattice_constant: float, index: int):
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from catalyst.graph.alignnd import alignn_gen

    atoms = bulk("Al", "fcc", a=float(lattice_constant), cubic=True)
    atoms.pbc = True
    atoms.calc = EMT()

    energy_per_atom = float(atoms.get_potential_energy()) / len(atoms)
    volume_per_atom = float(atoms.get_volume()) / len(atoms)

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
            "auto_retry_graph": True,
            "max_graph_attempts": 4,
            "require_bonds": True,
            "require_angles": False,
            "require_dihedrals": False,
            "retry_verbose": False,
        }
    )

    graph.gid = f"al_fcc_multiscalar_{index}"
    graph.target_scalars = torch.tensor(
        [[energy_per_atom, volume_per_atom, float(lattice_constant)]],
        dtype=torch.float32,
    )
    graph.y = graph.target_scalars.clone()
    return graph


def main() -> None:
    from torch_geometric.loader import DataLoader

    torch.manual_seed(7)

    task = GNNTask.graph_multiscalar(
        num_targets=3,
        target_key="target_scalars",
        target_names=["energy_per_atom", "volume_per_atom", "lattice_constant"],
    )

    parameters = {"model_dict": {"prediction_params": {}}}
    task.apply_to_catalyst_parameters(parameters)

    model = build_task_model(
        task=task,
        preset="alignn",
        num_species=1,
        cutoff=3.5,
        dim=24,
        num_convs=1,
        act=nn.SiLU(),
    )

    assert isinstance(model, GraphMultiScalarAdapter)
    assert isinstance(model.model.decoder, MultiScalarDecoder)
    assert model.model.decoder.num_targets == 3

    graphs = [
        make_al_graph(3.98, 0),
        make_al_graph(4.05, 1),
    ]

    follow_batch = ["x_atm", "x_bnd"]
    if graphs[0].x_ang is not None:
        follow_batch.append("x_ang")

    batch = next(iter(DataLoader(graphs, batch_size=2, follow_batch=follow_batch)))
    validate_task_batch(task=task, model=model, batch=batch, print_summary=True)

    with torch.no_grad():
        prediction = model(batch)

    assert tuple(prediction.shape) == (2, 3)
    assert tuple(batch.target_scalars.shape) == (2, 3)
    assert torch.isfinite(prediction).all()

    print("04_al_fcc_alignn_graph_multiscalar_smoke passed.")


if __name__ == "__main__":
    main()
