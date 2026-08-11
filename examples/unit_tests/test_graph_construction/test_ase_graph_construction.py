"""Scientific regression tests for ASE -> Catalyst atomic graph construction."""

from __future__ import annotations

import math

import torch
from ase.build import bulk

from catalyst.graph.alignnd import alignn_gen


def _alignn_graph(atoms, *, include_angs=True):
    return alignn_gen(
        {
            "type": "alignnd",
            "raw_data": atoms.copy(),
            "neighbor_params": [3.2, 12],
            "include_angs": include_angs,
            "is_dihedral": False,
            "store_raw_data": False,
            "use_pt": False,
            "include_equivariant_fields": True,
            "auto_retry_graph": False,
            "require_bonds": True,
            "require_angles": include_angs,
            "require_dihedrals": False,
            "retry_verbose": False,
        }
    )


def test_fcc_al_first_neighbor_graph_has_expected_coordination_and_distance():
    lattice_constant = 4.05
    atoms = bulk("Al", "fcc", a=lattice_constant, cubic=True).repeat((2, 2, 2))
    atoms.pbc = True

    graph = _alignn_graph(atoms, include_angs=True)

    n_atoms = len(atoms)
    assert graph.x_atm.shape[0] == n_atoms
    assert graph.edge_index_G.shape[0] == 2
    assert graph.edge_index_G.shape[1] == graph.x_bnd.numel()
    assert graph.edge_index_A is not None
    assert graph.x_ang is not None
    assert graph.edge_index_A.shape[1] == graph.x_ang.numel()

    # The radius is between the first and second FCC neighbor shells. Catalyst's
    # global bond graph stores each physical pair once, so counting each endpoint
    # should recover the FCC coordination number of 12 for every atom.
    src, dst = graph.edge_index_G
    coordination = torch.bincount(torch.cat([src, dst]), minlength=n_atoms)
    assert torch.equal(coordination, torch.full_like(coordination, 12))

    expected_distance = lattice_constant / math.sqrt(2.0)
    torch.testing.assert_close(
        graph.x_bnd.float(),
        torch.full_like(graph.x_bnd.float(), expected_distance),
        rtol=1.0e-4,
        atol=5.0e-4,
    )

    assert graph.pos.shape == (n_atoms, 3)
    assert graph.z.shape == (n_atoms,)
    assert graph.edge_index.shape == graph.edge_index_G.shape
    assert graph.shifts.shape == (graph.edge_index.size(1), 3)
    assert graph.edge_vec.shape == (graph.edge_index.size(1), 3)
    assert graph.edge_dist.shape[0] == graph.edge_index.size(1)
    torch.testing.assert_close(
        graph.edge_dist.reshape(-1),
        graph.x_bnd.float(),
        rtol=1.0e-4,
        atol=5.0e-4,
    )


def test_atomic_alignnd_returns_one_local_graph_per_atom():
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.pbc = True

    local_graphs = alignn_gen(
        {
            "type": "atomic_alignnd",
            "raw_data": atoms.copy(),
            "element_list": ["Al"],
            "neighbor_params": [3.2, 12],
            "include_angs": False,
            "is_dihedral": False,
            "store_raw_data": False,
            "use_pt": False,
            "cpu_cores": 1,
            "include_equivariant_fields": True,
            "auto_retry_graph": False,
            "require_bonds": True,
            "require_angles": False,
            "require_dihedrals": False,
            "retry_verbose": False,
        }
    )

    assert len(local_graphs) == len(atoms)
    for graph in local_graphs:
        assert graph.num_nodes >= 2
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.size(1) > 0
        assert graph.global_atom_indices is not None
        assert graph.global_atom_indices.numel() == graph.num_nodes
        assert torch.all(graph.global_atom_indices >= 0)
        assert torch.all(graph.global_atom_indices < len(atoms))


def test_realignnd_tracks_structure_boundaries_for_multiple_ase_structures():
    structures = []
    for lattice_constant in (3.98, 4.05, 4.12):
        atoms = bulk("Al", "fcc", a=lattice_constant, cubic=True)
        atoms.pbc = True
        structures.append(atoms)

    graph = alignn_gen(
        {
            "type": "realignnd",
            "raw_data": structures,
            "neighbor_params": [3.5, 12],
            "include_angs": False,
            "is_dihedral": False,
            "store_raw_data": False,
            "use_pt": False,
            "include_equivariant_fields": True,
            "auto_retry_graph": False,
            "require_bonds": True,
            "require_angles": False,
            "require_dihedrals": False,
            "retry_verbose": False,
        }
    )

    n_per_structure = len(structures[0])
    assert graph.x_atm.size(0) == 3 * n_per_structure
    assert graph.atm_amounts.tolist() == [
        n_per_structure,
        2 * n_per_structure,
        3 * n_per_structure,
    ]

    assert graph.cell.shape == (3, 3, 3)
    assert graph.pbc.shape == (3, 3)
    assert graph.atom_graph_batch.shape[0] == graph.pos.shape[0]
    counts = torch.bincount(graph.atom_graph_batch, minlength=3)
    assert counts.tolist() == [n_per_structure, n_per_structure, n_per_structure]
    assert graph.edge_graph_batch.shape[0] == graph.edge_index.size(1)
