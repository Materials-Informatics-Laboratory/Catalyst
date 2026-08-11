"""Scientific invariance/equivariance regression tests for Catalyst's EGNN route."""

from __future__ import annotations

import copy

import torch
from ase.build import bulk

from catalyst.graph.alignnd import alignn_gen
from catalyst.ml.gnn import GNNTask, build_task_model


def _make_asymmetric_al_graph():
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 2))
    atoms.pbc = True

    # Break perfect FCC inversion symmetry so a random vector readout is not
    # trivially zero everywhere.
    atoms.positions[0] += [0.08, -0.03, 0.05]

    return alignn_gen(
        {
            "type": "alignnd",
            "raw_data": atoms,
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


def _rotate_graph(graph, rotation):
    rotated = copy.deepcopy(graph)
    rotation = rotation.to(dtype=rotated.pos.dtype, device=rotated.pos.device)

    # Catalyst stores Cartesian vectors as row vectors.
    rotated.pos = rotated.pos @ rotation.T
    rotated.cell = rotated.cell @ rotation.T

    if getattr(rotated, "edge_vec", None) is not None:
        rotated.edge_vec = rotated.edge_vec @ rotation.T

    return rotated


def _translate_graph(graph, translation):
    translated = copy.deepcopy(graph)
    translation = translation.to(dtype=translated.pos.dtype, device=translated.pos.device)
    translated.pos = translated.pos + translation
    return translated


def test_equivariant_node_vector_rotates_and_is_translation_invariant():
    torch.manual_seed(19)
    graph = _make_asymmetric_al_graph()

    task = GNNTask.node_vector(target_key="target_vector", vector_channels=1)
    model = build_task_model(
        task=task,
        model_type="equivariant",
        return_dict=False,
        num_species=1,
        cutoff=3.5,
        dim=20,
        num_convs=2,
    )
    model.eval()

    rotation = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        original = model(copy.deepcopy(graph))
        rotated = model(_rotate_graph(graph, rotation))
        translated = model(_translate_graph(graph, torch.tensor([0.37, -0.22, 0.41])))

    expected_rotated = original @ rotation.T

    # Avoid a vacuous all-zero equivariance check on a highly symmetric system.
    assert float(torch.linalg.norm(original)) > 1.0e-8
    torch.testing.assert_close(rotated, expected_rotated, rtol=2.0e-4, atol=2.0e-5)
    torch.testing.assert_close(translated, original, rtol=2.0e-4, atol=2.0e-5)


def test_equivariant_graph_scalar_is_rotation_and_translation_invariant():
    torch.manual_seed(23)
    graph = _make_asymmetric_al_graph()

    task = GNNTask.graph_scalar(target_key="target_scalar")
    model = build_task_model(
        task=task,
        model_type="equivariant",
        return_dict=False,
        num_species=1,
        cutoff=3.5,
        dim=20,
        num_convs=2,
    )
    model.eval()

    rotation = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        original = model(copy.deepcopy(graph))
        rotated = model(_rotate_graph(graph, rotation))
        translated = model(_translate_graph(graph, torch.tensor([-0.15, 0.28, 0.33])))

    torch.testing.assert_close(rotated, original, rtol=2.0e-4, atol=2.0e-5)
    torch.testing.assert_close(translated, original, rtol=2.0e-4, atol=2.0e-5)
