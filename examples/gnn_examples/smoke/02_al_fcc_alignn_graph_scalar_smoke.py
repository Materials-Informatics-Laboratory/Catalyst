"""
Smoke example 02: Al FCC ALIGNN graph_scalar task/preset passthrough.

This smoke test exercises the important passthrough route:

    build_task_model(
        task=GNNTask.graph_scalar(...),
        preset="alignn",
        apply_task_model_kwargs=False,
        decoder=CustomReadout(...),
        ...
    )

The task configures Catalyst's backend contract, while the ALIGNN preset/custom
decoder architecture is preserved.

Run:
    python examples/gnn_examples/smoke/02_al_fcc_alignn_graph_scalar_smoke.py
"""

from __future__ import annotations

import torch
from torch import nn

from catalyst.ml.gnn.tasks import GNNTask, build_task_model, validate_task_batch


class SmokeAlignnGraphScalarReadout(nn.Module):
    def __init__(self, dim: int, act=nn.SiLU()):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            act,
            nn.Linear(dim, 1),
        )

    def _first(self, data, *names):
        for name in names:
            if hasattr(data, name) and getattr(data, name) is not None:
                return getattr(data, name)
        raise AttributeError(f"Missing one of {names}")

    def forward(self, data):
        h = self._first(data, "h_atm", "h_1", "h_scalar")
        batch = getattr(
            data,
            "x_atm_batch",
            getattr(data, "x_1_batch", getattr(data, "batch", None)),
        )
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        local = self.head(h).view(-1, 1)
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        out = local.new_zeros((n_graphs, 1))
        out.index_add_(0, batch.to(local.device), local)
        return out.view(-1)


def make_al_graph():
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from catalyst.graph.alignnd import alignn_gen

    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((1, 1, 1))
    atoms.pbc = True
    atoms.calc = EMT()
    energy = float(atoms.get_potential_energy())

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

    graph.gid = "al_fcc_smoke_graph_scalar"
    graph.energy_eV = torch.tensor([energy], dtype=torch.float32)
    graph.target_scalar = torch.tensor([0.0], dtype=torch.float32)
    graph.y = graph.target_scalar.clone()
    return graph


def main() -> None:
    from torch_geometric.loader import DataLoader

    task = GNNTask.graph_scalar(target_key="target_scalar")

    parameters = {"model_dict": {"prediction_params": {}}}
    task.apply_to_catalyst_parameters(parameters)

    model = build_task_model(
        task=task,
        preset="alignn",
        apply_task_model_kwargs=False,
        conv_type="gated_gcn",
        processor_type="order",
        decoder=SmokeAlignnGraphScalarReadout(dim=32),
        num_species=1,
        cutoff=3.5,
        dim=32,
        num_convs=1,
        act=nn.SiLU(),
        encode_3body=True,
        dihedral=False,
    )

    graph = make_al_graph()
    follow_batch = ["x_atm", "x_bnd"]
    if hasattr(graph, "x_ang") and graph.x_ang is not None:
        follow_batch.append("x_ang")

    batch = next(iter(DataLoader([graph], batch_size=1, follow_batch=follow_batch)))
    validate_task_batch(task=task, model=model, batch=batch, print_summary=True)

    print("02_al_fcc_alignn_graph_scalar_smoke passed.")


if __name__ == "__main__":
    main()
