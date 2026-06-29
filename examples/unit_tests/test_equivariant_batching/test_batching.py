#!/usr/bin/env python
"""
Batch test: build Al FCC equivariant ALIGNN graphs, then PyG-batch them.

This is a self-contained test script. It does not load precomputed graphs.
Instead it:

1. Builds an Al FCC supercell with ASE.
2. Runs a very short EMT MD trajectory to create several frames.
3. Converts each frame to an updated ALIGNN graph with equivariant fields.
4. Batches the graphs with torch_geometric.data.Batch.
5. Validates that equivariant fields batch correctly.

Expected graph fields
---------------------
Each graph should contain:

    z
    pos
    edge_index
    cell
    pbc
    shifts
    num_nodes

Usually it will also contain:

    edge_vec
    edge_dist

and the legacy ALIGNN fields:

    x_atm
    x_bnd
    x_ang
    edge_index_G
    edge_index_A
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from torch_geometric.data import Batch

from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet


# =============================================================================
# Catalyst import helpers
# =============================================================================


def import_alignn_gen():
    """Import alignn_gen from common Catalyst layouts."""
    try:
        from catalyst.graph.alignnd import alignn_gen
        return alignn_gen
    except ImportError:
        pass

    try:
        from catalyst.src.graph.alignnd import alignn_gen
        return alignn_gen
    except ImportError as exc:
        raise ImportError(
            "Could not import alignn_gen. Tried:\n"
            "  catalyst.graph.alignnd\n"
            "  catalyst.src.graph.alignnd\n\n"
            "Make sure Catalyst is installed in editable mode or that your "
            "package source directory is on PYTHONPATH."
        ) from exc


def import_equivariant_graph_helpers():
    """Import optional fallback helpers from graph.py."""
    try:
        from catalyst.graph.graph import build_equivariant_atomic_fields, attach_equivariant_fields
        return build_equivariant_atomic_fields, attach_equivariant_fields
    except ImportError:
        pass

    try:
        from catalyst.src.graph.graph import build_equivariant_atomic_fields, attach_equivariant_fields
        return build_equivariant_atomic_fields, attach_equivariant_fields
    except ImportError:
        return None, None


# =============================================================================
# Structure / MD
# =============================================================================


def build_al_fcc(
    *,
    lattice_constant: float = 4.05,
    repeat: Tuple[int, int, int] = (2, 2, 2),
):
    atoms = bulk("Al", "fcc", a=lattice_constant, cubic=True)
    atoms = atoms.repeat(repeat)
    atoms.pbc = True
    atoms.calc = EMT()
    return atoms


def make_md_frames(
    atoms,
    *,
    n_frames: int = 4,
    md_steps_per_frame: int = 2,
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
    seed: int = 7,
):
    """Run tiny EMT MD and return sampled frames."""
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1.")

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = VelocityVerlet(atoms, timestep_fs * units.fs)

    frames = [atoms.copy()]
    frames[0].calc = EMT()

    for _ in range(1, n_frames):
        dyn.run(md_steps_per_frame)
        frame = atoms.copy()
        frame.calc = EMT()
        frames.append(frame)

    return frames


# =============================================================================
# Graph generation and metadata finalization
# =============================================================================


def finalize_graph_metadata(graph):
    """Make graph.num_nodes and key aliases explicit."""
    z = getattr(graph, "z", None)
    pos = getattr(graph, "pos", None)
    x_atm = getattr(graph, "x_atm", None)

    if z is not None:
        graph.num_nodes = int(z.size(0))
    elif pos is not None:
        graph.num_nodes = int(pos.size(0))
    elif x_atm is not None:
        graph.num_nodes = int(x_atm.size(0))

    if getattr(graph, "edge_index", None) is None and getattr(graph, "edge_index_G", None) is not None:
        graph.edge_index = graph.edge_index_G

    edge_index = getattr(graph, "edge_index", None)
    if getattr(graph, "shifts", None) is None and edge_index is not None:
        graph.shifts = torch.zeros((edge_index.size(1), 3), dtype=torch.long, device=edge_index.device)

    return graph


def ensure_equivariant_fields(graph, atoms):
    """Attach equivariant fields if alignn_gen did not do it directly."""
    required = ("z", "pos", "edge_index", "cell", "pbc", "shifts")
    if all(getattr(graph, key, None) is not None for key in required):
        return finalize_graph_metadata(graph)

    build_equivariant_atomic_fields, attach_equivariant_fields = import_equivariant_graph_helpers()

    if build_equivariant_atomic_fields is None or attach_equivariant_fields is None:
        missing = [key for key in required if getattr(graph, key, None) is None]
        raise RuntimeError(
            "Graph is missing equivariant fields and fallback helpers could not be imported. "
            f"Missing fields: {missing}"
        )

    if getattr(graph, "edge_index_G", None) is None:
        raise RuntimeError("Cannot attach fallback equivariant fields because graph.edge_index_G is missing.")

    fields = build_equivariant_atomic_fields(
        atoms,
        graph.edge_index_G.detach().cpu().numpy(),
        dtype=np.float32,
        include_edge_geometry=True,
    )
    graph = attach_equivariant_fields(graph, **fields)
    return finalize_graph_metadata(graph)


def build_graphs_from_frames(
    frames: Sequence,
    *,
    cutoff: float = 5.0,
    k: int = -1,
    include_angs: bool = True,
    include_dihedrals: bool = False,
    retry_verbose: bool = False,
):
    """Build updated ALIGNN graphs with equivariant fields from ASE frames."""
    alignn_gen = import_alignn_gen()

    graphs = []
    for frame_idx, atoms in enumerate(frames):
        graph = alignn_gen(
            {
                "type": "alignnd",
                "raw_data": atoms,
                "neighbor_params": [float(cutoff), int(k)],
                "include_angs": bool(include_angs),
                "is_dihedral": bool(include_dihedrals),

                # New equivariant-field controls.
                "include_equivariant_fields": True,
                "include_edge_geometry": True,

                # Keep this quiet by default.
                "retry_verbose": bool(retry_verbose),

                # For this FCC example these should succeed cleanly.
                "require_bonds": True,
                "require_angles": bool(include_angs),
                "require_dihedrals": False,
            }
        )

        graph = ensure_equivariant_fields(graph, atoms)
        validate_single_graph(graph, name=f"graph[{frame_idx}]")
        graphs.append(graph)

    return graphs


# =============================================================================
# Validation
# =============================================================================


def _shape(value):
    if value is None:
        return None
    if hasattr(value, "shape"):
        return tuple(value.shape)
    return type(value).__name__


def validate_single_graph(graph, *, name: str = "graph"):
    """Validate one graph before batching."""
    required = ("z", "pos", "edge_index", "cell", "pbc", "shifts", "num_nodes")
    missing = [key for key in required if getattr(graph, key, None) is None]
    if missing:
        raise AssertionError(f"{name} missing required fields: {missing}")

    n_nodes = int(graph.num_nodes)
    n_edges = int(graph.edge_index.size(1))

    assert graph.z.dim() == 1, f"{name}: z should be [N], got {_shape(graph.z)}"
    assert graph.z.size(0) == n_nodes, f"{name}: z length != num_nodes"

    assert graph.pos.dim() == 2 and graph.pos.size(1) == 3, f"{name}: pos should be [N, 3]"
    assert graph.pos.size(0) == n_nodes, f"{name}: pos length != num_nodes"

    assert graph.edge_index.dim() == 2 and graph.edge_index.size(0) == 2, (
        f"{name}: edge_index should be [2, E], got {_shape(graph.edge_index)}"
    )

    if n_edges > 0:
        assert int(graph.edge_index.min()) >= 0, f"{name}: negative edge index"
        assert int(graph.edge_index.max()) < n_nodes, f"{name}: edge index outside node range"

    assert graph.shifts.dim() == 2 and graph.shifts.size(1) == 3, (
        f"{name}: shifts should be [E, 3], got {_shape(graph.shifts)}"
    )
    assert graph.shifts.size(0) == n_edges, f"{name}: shifts length != n_edges"

    if getattr(graph, "edge_vec", None) is not None:
        assert graph.edge_vec.dim() == 2 and graph.edge_vec.size(1) == 3, (
            f"{name}: edge_vec should be [E, 3], got {_shape(graph.edge_vec)}"
        )
        assert graph.edge_vec.size(0) == n_edges, f"{name}: edge_vec length != n_edges"

    if getattr(graph, "edge_dist", None) is not None:
        assert graph.edge_dist.size(0) == n_edges, f"{name}: edge_dist length != n_edges"

    # Legacy ALIGNN fields should still be present.
    assert getattr(graph, "edge_index_G", None) is not None, f"{name}: missing edge_index_G"
    assert getattr(graph, "x_atm", None) is not None, f"{name}: missing x_atm"
    assert getattr(graph, "x_bnd", None) is not None, f"{name}: missing x_bnd"

    return True


def recompute_batched_edge_geometry(batch):
    """Recompute edge vectors/distances from batched pos/cell/shifts.

    This mimics what an EquivariantProcessor should do for conservative
    energy-gradient force models.
    """
    src, dst = batch.edge_index

    if getattr(batch, "batch", None) is None:
        raise AssertionError("Batched graph is missing batch.batch.")

    edge_batch = batch.batch[src]

    if getattr(batch, "cell", None) is None:
        shift_vec = torch.zeros_like(batch.pos[dst])
    else:
        cell = batch.cell[edge_batch]
        shifts = batch.shifts.to(device=batch.pos.device, dtype=batch.pos.dtype)
        shift_vec = torch.einsum("ei,eij->ej", shifts, cell)

    edge_vec = batch.pos[dst] + shift_vec - batch.pos[src]
    edge_dist = torch.linalg.norm(edge_vec, dim=-1)

    return edge_vec, edge_dist


def validate_batched_graph(batch, graphs: Sequence):
    """Validate Batch.from_data_list(graphs)."""
    expected_nodes = sum(int(graph.num_nodes) for graph in graphs)
    expected_edges = sum(int(graph.edge_index.size(1)) for graph in graphs)

    assert batch.num_graphs == len(graphs), "batch.num_graphs mismatch"
    assert batch.pos.size(0) == expected_nodes, "batched pos length mismatch"
    assert batch.z.size(0) == expected_nodes, "batched z length mismatch"
    assert batch.edge_index.size(1) == expected_edges, "batched edge count mismatch"
    assert batch.shifts.size(0) == expected_edges, "batched shifts length mismatch"

    assert getattr(batch, "batch", None) is not None, "batch.batch was not created"
    assert batch.batch.size(0) == expected_nodes, "batch.batch length mismatch"
    assert getattr(batch, "ptr", None) is not None, "batch.ptr was not created"
    assert batch.ptr.numel() == len(graphs) + 1, "batch.ptr length mismatch"

    if expected_edges > 0:
        assert int(batch.edge_index.min()) >= 0, "batched edge_index has negative values"
        assert int(batch.edge_index.max()) < expected_nodes, "batched edge_index exceeds node count"

    # graph.py should stack cell and pbc as graph-level attributes.
    assert batch.cell.dim() == 3 and batch.cell.shape[-2:] == (3, 3), (
        f"Expected batch.cell shape [B, 3, 3], got {tuple(batch.cell.shape)}"
    )
    assert batch.cell.size(0) == len(graphs), "batch.cell first dimension should equal number of graphs"

    assert batch.pbc.dim() == 2 and batch.pbc.shape[-1] == 3, (
        f"Expected batch.pbc shape [B, 3], got {tuple(batch.pbc.shape)}"
    )
    assert batch.pbc.size(0) == len(graphs), "batch.pbc first dimension should equal number of graphs"

    recomputed_edge_vec, recomputed_edge_dist = recompute_batched_edge_geometry(batch)

    assert recomputed_edge_vec.shape == (expected_edges, 3), "recomputed edge_vec shape mismatch"
    assert recomputed_edge_dist.shape == (expected_edges,), "recomputed edge_dist shape mismatch"

    # If precomputed edge geometry exists, compare it to the recomputed geometry.
    # This should be close if shifts/cell/pos conventions are consistent.
    if getattr(batch, "edge_vec", None) is not None and batch.edge_vec.size(0) == expected_edges:
        max_vec_error = torch.max(torch.abs(batch.edge_vec - recomputed_edge_vec)).item()
    else:
        max_vec_error = float("nan")

    if getattr(batch, "edge_dist", None) is not None and batch.edge_dist.size(0) == expected_edges:
        batch_edge_dist = batch.edge_dist.reshape(-1).to(recomputed_edge_dist.device)
        max_dist_error = torch.max(torch.abs(batch_edge_dist - recomputed_edge_dist)).item()
    else:
        max_dist_error = float("nan")

    return {
        "expected_nodes": expected_nodes,
        "expected_edges": expected_edges,
        "max_edge_vec_error": max_vec_error,
        "max_edge_dist_error": max_dist_error,
    }


# =============================================================================
# Reporting / optional saving
# =============================================================================


def print_single_graph_summary(graphs: Sequence):
    print("\nSingle-graph summaries")
    print("----------------------")
    for idx, graph in enumerate(graphs):
        n_nodes = int(graph.num_nodes)
        n_edges = int(graph.edge_index.size(1))
        n_alignn_edges = int(graph.edge_index_G.size(1)) if getattr(graph, "edge_index_G", None) is not None else -1
        n_angle_edges = int(graph.edge_index_A.size(1)) if getattr(graph, "edge_index_A", None) is not None else 0

        print(
            f"graph[{idx}] "
            f"nodes={n_nodes:4d} "
            f"equivariant_edges={n_edges:5d} "
            f"alignn_edges={n_alignn_edges:5d} "
            f"angle_edges={n_angle_edges:6d} "
            f"cell={tuple(graph.cell.shape)} "
            f"pbc={tuple(graph.pbc.shape)}"
        )


def print_batch_summary(batch, validation_info):
    print("\nBatched graph summary")
    print("---------------------")
    print(f"num_graphs:       {batch.num_graphs}")
    print(f"z:                {tuple(batch.z.shape)}")
    print(f"pos:              {tuple(batch.pos.shape)}")
    print(f"edge_index:       {tuple(batch.edge_index.shape)}")
    print(f"shifts:           {tuple(batch.shifts.shape)}")
    print(f"cell:             {tuple(batch.cell.shape)}")
    print(f"pbc:              {tuple(batch.pbc.shape)}")
    print(f"batch:            {tuple(batch.batch.shape)}")
    print(f"ptr:              {tuple(batch.ptr.shape)}")
    print(f"expected_nodes:   {validation_info['expected_nodes']}")
    print(f"expected_edges:   {validation_info['expected_edges']}")
    print(f"max edge_vec err: {validation_info['max_edge_vec_error']:.6e}")
    print(f"max edge_dist err:{validation_info['max_edge_dist_error']:.6e}")


def save_outputs(graphs: Sequence, batch, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, graph in enumerate(graphs):
        torch.save(graph, output_dir / f"graph_{idx:04d}.pt")

    torch.save(batch, output_dir / "batch.pt")


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Al FCC equivariant ALIGNN graphs and validate PyG batching."
    )

    parser.add_argument("--repeat", type=int, nargs=3, default=(2, 2, 2))
    parser.add_argument("--lattice-constant", type=float, default=4.05)

    parser.add_argument("--n-frames", type=int, default=4)
    parser.add_argument("--md-steps-per-frame", type=int, default=2)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--k", type=int, default=-1)
    parser.add_argument("--include-angs", action="store_true", default=True)
    parser.add_argument("--no-include-angs", dest="include_angs", action="store_false")
    parser.add_argument("--include-dihedrals", action="store_true")
    parser.add_argument("--retry-verbose", action="store_true")

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("al_fcc_equivariant_batch_test"))

    return parser.parse_args()


def main():
    args = parse_args()

    atoms = build_al_fcc(
        lattice_constant=args.lattice_constant,
        repeat=tuple(args.repeat),
    )

    print(f"Built Al FCC supercell with {len(atoms)} atoms.")
    print(f"Generating {args.n_frames} MD-sampled frames...")

    frames = make_md_frames(
        atoms,
        n_frames=args.n_frames,
        md_steps_per_frame=args.md_steps_per_frame,
        temperature_K=args.temperature_K,
        timestep_fs=args.timestep_fs,
        seed=args.seed,
    )

    print("Building equivariant ALIGNN graphs...")
    graphs = build_graphs_from_frames(
        frames,
        cutoff=args.cutoff,
        k=args.k,
        include_angs=args.include_angs,
        include_dihedrals=args.include_dihedrals,
        retry_verbose=args.retry_verbose,
    )

    print_single_graph_summary(graphs)

    print("\nBatching graphs with torch_geometric.data.Batch.from_data_list...")
    batch = Batch.from_data_list(graphs)
    validation_info = validate_batched_graph(batch, graphs)

    print_batch_summary(batch, validation_info)

    if args.save:
        save_outputs(graphs, batch, args.output_dir)
        print(f"\nSaved graphs and batch to: {args.output_dir}")

    print("\nPASS: equivariant ALIGNN graph batching is valid.")


if __name__ == "__main__":
    main()
