#!/usr/bin/env python
"""
Example: Al FCC MD -> equivariant ALIGNN graphs -> NetworkX plots.

This script demonstrates the updated graph-builder ecosystem where ALIGNN-style
graphs can also carry standardized equivariant fields:

    z, pos, edge_index, edge_vec, edge_dist, cell, pbc, shifts, num_nodes

Workflow
--------
1. Build an Al FCC supercell with ASE.
2. Run short MD using ASE EMT.
3. Convert sampled MD frames to ALIGNN graphs with equivariant fields.
4. Validate the graph fields needed by future equivariant GNNs.
5. Save each graph and make NetworkX visualizations.

Recommended location
--------------------
examples/graphs/al_fcc_md_equivariant_alignnd_networkx.py

Notes
-----
- EMT is used only as a lightweight demonstration calculator.
- The NetworkX plot is a 2D projection of the equivariant atomic neighbor graph.
- For future conservative energy/force models, an EquivariantProcessor should
  recompute edge_vec and edge_dist from pos/cell/shifts inside forward() so
  autograd tracks positions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib.pyplot as plt
import networkx as nx

from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import Trajectory, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet


# =============================================================================
# Catalyst imports with compatibility fallbacks
# =============================================================================


def import_alignn_gen():
    """Import alignn_gen from either the modern or legacy Catalyst path."""
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
            "  catalyst.src.graph.alignnd\n"
            "Make sure Catalyst is installed in editable mode or that the package "
            "root is on PYTHONPATH."
        ) from exc


def import_equivariant_graph_helpers():
    """Import optional graph.py helper functions.

    These helpers are used as a fallback if alignn_gen does not directly attach
    equivariant fields. If your updated alignnd.py already attaches them, this
    fallback will not change anything.
    """
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
# MD generation
# =============================================================================


def build_al_fcc_supercell(
    lattice_constant: float = 4.05,
    repeat: Tuple[int, int, int] = (3, 3, 3),
):
    """Build a periodic Al FCC supercell."""
    atoms = bulk("Al", "fcc", a=lattice_constant, cubic=True)
    atoms = atoms.repeat(repeat)
    atoms.pbc = True
    atoms.calc = EMT()
    return atoms


def run_md(
    atoms,
    *,
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
    n_steps: int = 100,
    sample_interval: int = 10,
    seed: int = 7,
    trajectory_path: Optional[Path] = None,
):
    """Run a simple NVE MD trajectory and return sampled frames plus metadata."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = VelocityVerlet(atoms, timestep_fs * units.fs)

    frames = []
    rows = []

    if trajectory_path is not None:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        traj = Trajectory(str(trajectory_path), "w", atoms)
    else:
        traj = None

    def sample(step: int):
        frame = atoms.copy()
        frame.calc = EMT()

        epot = float(atoms.get_potential_energy())
        ekin = float(atoms.get_kinetic_energy())
        temp = float(atoms.get_temperature())

        frames.append(frame)
        rows.append(
            {
                "step": int(step),
                "time_fs": float(step * timestep_fs),
                "potential_energy_eV": epot,
                "kinetic_energy_eV": ekin,
                "total_energy_eV": epot + ekin,
                "temperature_K": temp,
            }
        )

        if traj is not None:
            traj.write(atoms)

    sample(0)
    for step in range(1, n_steps + 1):
        dyn.run(1)
        if step % sample_interval == 0 or step == n_steps:
            sample(step)

    if traj is not None:
        traj.close()

    return frames, rows


# =============================================================================
# Graph generation
# =============================================================================


def finalize_graph_metadata(graph):
    """Make sure PyG/equivariant metadata is explicit."""
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
    """Ensure a graph has the standardized equivariant fields.

    The updated alignnd.py should attach these fields directly. This fallback
    uses graph.py helpers when available.
    """
    has_required = all(
        getattr(graph, key, None) is not None
        for key in ("z", "pos", "edge_index", "cell", "pbc", "shifts")
    )

    if has_required:
        return finalize_graph_metadata(graph)

    build_equivariant_atomic_fields, attach_equivariant_fields = import_equivariant_graph_helpers()

    if build_equivariant_atomic_fields is None or attach_equivariant_fields is None:
        missing = [
            key for key in ("z", "pos", "edge_index", "cell", "pbc", "shifts")
            if getattr(graph, key, None) is None
        ]
        raise RuntimeError(
            "The graph is missing equivariant fields and graph.py fallback helpers "
            f"could not be imported. Missing fields: {missing}"
        )

    if getattr(graph, "edge_index_G", None) is None:
        raise RuntimeError("Cannot attach equivariant fields because graph.edge_index_G is missing.")

    edge_index_np = graph.edge_index_G.detach().cpu().numpy()
    fields = build_equivariant_atomic_fields(
        atoms,
        edge_index_np,
        dtype=np.float32,
        include_edge_geometry=True,
    )
    graph = attach_equivariant_fields(graph, **fields)
    return finalize_graph_metadata(graph)


def build_equivariant_alignnd_graphs(
    frames: Sequence,
    *,
    cutoff: float = 5.0,
    k: int = -1,
    include_angs: bool = True,
    include_dihedrals: bool = False,
    store_raw_data: bool = False,
    use_periodic_table_basis: bool = False,
    retry_verbose: bool = False,
):
    """Convert MD frames into updated ALIGNN graphs carrying equivariant fields."""
    alignn_gen = import_alignn_gen()

    graphs = []
    for frame_index, atoms in enumerate(frames):
        graph = alignn_gen(
            {
                "type": "alignnd",
                "raw_data": atoms,
                "neighbor_params": [float(cutoff), int(k)],
                "is_dihedral": bool(include_dihedrals),
                "store_raw_data": bool(store_raw_data),
                "use_pt": bool(use_periodic_table_basis),
                "include_angs": bool(include_angs),

                # New equivariant-field controls from the updated graph builders.
                "include_equivariant_fields": True,
                "include_edge_geometry": True,

                # Keep the retry system quiet for an example unless requested.
                "retry_verbose": bool(retry_verbose),
                "require_bonds": True,
                "require_angles": bool(include_angs),
                "require_dihedrals": False,
            }
        )

        graph = ensure_equivariant_fields(graph, atoms)
        validate_equivariant_graph(graph, name=f"frame {frame_index}")
        graphs.append(graph)

    return graphs


def validate_equivariant_graph(graph, *, name: str = "graph"):
    """Validate the equivariant graph contract."""
    required = ("z", "pos", "edge_index", "cell", "pbc", "shifts", "num_nodes")
    missing = [key for key in required if getattr(graph, key, None) is None]
    if missing:
        raise ValueError(f"{name} is missing required fields: {missing}")

    n_nodes = int(graph.num_nodes)
    n_edges = int(graph.edge_index.size(1))

    if graph.z.size(0) != n_nodes:
        raise ValueError(f"{name}: z has {graph.z.size(0)} rows but num_nodes={n_nodes}.")

    if graph.pos.size(0) != n_nodes or graph.pos.size(1) != 3:
        raise ValueError(f"{name}: pos must have shape [num_nodes, 3], got {tuple(graph.pos.shape)}.")

    if graph.edge_index.dim() != 2 or graph.edge_index.size(0) != 2:
        raise ValueError(f"{name}: edge_index must have shape [2, n_edges], got {tuple(graph.edge_index.shape)}.")

    if n_edges > 0:
        if int(graph.edge_index.min()) < 0 or int(graph.edge_index.max()) >= n_nodes:
            raise ValueError(f"{name}: edge_index contains node ids outside [0, {n_nodes - 1}].")

    if graph.shifts.size(0) != n_edges or graph.shifts.size(1) != 3:
        raise ValueError(
            f"{name}: shifts must have shape [n_edges, 3], got {tuple(graph.shifts.shape)}."
        )

    if getattr(graph, "edge_vec", None) is not None and graph.edge_vec.size(0) != n_edges:
        raise ValueError(f"{name}: edge_vec length does not match n_edges.")

    if getattr(graph, "edge_dist", None) is not None and graph.edge_dist.size(0) != n_edges:
        raise ValueError(f"{name}: edge_dist length does not match n_edges.")

    return True


# =============================================================================
# NetworkX plotting
# =============================================================================


def graph_to_networkx(graph, *, undirected: bool = True):
    """Convert the equivariant edge_index field to a NetworkX graph."""
    graph_cls = nx.Graph if undirected else nx.DiGraph
    nx_graph = graph_cls()

    n_nodes = int(graph.num_nodes)
    nx_graph.add_nodes_from(range(n_nodes))

    edge_index = graph.edge_index.detach().cpu().numpy()
    if getattr(graph, "edge_dist", None) is not None:
        edge_dist = graph.edge_dist.detach().cpu().numpy().reshape(-1)
    else:
        edge_dist = np.full(edge_index.shape[1], np.nan)

    for edge_id, (src, dst) in enumerate(edge_index.T):
        nx_graph.add_edge(int(src), int(dst), edge_id=int(edge_id), distance=float(edge_dist[edge_id]))

    return nx_graph


def get_projected_positions(graph, axes: Tuple[int, int] = (0, 1)):
    """Use the equivariant pos field as a 2D NetworkX layout."""
    pos = graph.pos.detach().cpu().numpy()
    ax0, ax1 = axes
    return {idx: (float(pos[idx, ax0]), float(pos[idx, ax1])) for idx in range(pos.shape[0])}


def plot_networkx_graph(
    graph,
    *,
    output_path: Path,
    title: str,
    axes: Tuple[int, int] = (0, 1),
    undirected: bool = True,
    node_size: float = 80.0,
    edge_width: float = 0.8,
):
    """Plot a single equivariant graph using NetworkX."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nx_graph = graph_to_networkx(graph, undirected=undirected)
    layout = get_projected_positions(graph, axes=axes)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    nx.draw_networkx_edges(nx_graph, layout, ax=ax, width=edge_width, alpha=0.65)
    nx.draw_networkx_nodes(nx_graph, layout, ax=ax, node_size=node_size)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_graph_montage(
    graphs: Sequence,
    rows: Sequence[dict],
    *,
    output_path: Path,
    frame_indices: Optional[Sequence[int]] = None,
    axes: Tuple[int, int] = (0, 1),
):
    """Plot several sampled graphs in one figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frame_indices is None:
        if len(graphs) <= 4:
            frame_indices = list(range(len(graphs)))
        else:
            frame_indices = np.linspace(0, len(graphs) - 1, 4, dtype=int).tolist()

    n = len(frame_indices)
    fig, axes_arr = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes_arr = [axes_arr]

    for ax, frame_idx in zip(axes_arr, frame_indices):
        graph = graphs[frame_idx]
        row = rows[frame_idx]

        nx_graph = graph_to_networkx(graph, undirected=True)
        layout = get_projected_positions(graph, axes=axes)

        nx.draw_networkx_edges(nx_graph, layout, ax=ax, width=0.6, alpha=0.6)
        nx.draw_networkx_nodes(nx_graph, layout, ax=ax, node_size=45)

        ax.set_title(f"step {row['step']}, {row['time_fs']:.0f} fs")
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def plot_md_summary(rows: Sequence[dict], *, output_path: Path):
    """Plot potential, kinetic, total energy, and temperature summaries."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time_fs = np.asarray([row["time_fs"] for row in rows], dtype=float)
    potential = np.asarray([row["potential_energy_eV"] for row in rows], dtype=float)
    kinetic = np.asarray([row["kinetic_energy_eV"] for row in rows], dtype=float)
    total = np.asarray([row["total_energy_eV"] for row in rows], dtype=float)
    temperature = np.asarray([row["temperature_K"] for row in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    ax1.plot(time_fs, potential, label="Potential energy")
    ax1.plot(time_fs, kinetic, label="Kinetic energy")
    ax1.plot(time_fs, total, label="Total energy")
    ax1.set_xlabel("Time (fs)")
    ax1.set_ylabel("Energy (eV)")

    ax2 = ax1.twinx()
    ax2.plot(time_fs, temperature, linestyle="--", label="Temperature")
    ax2.set_ylabel("Temperature (K)")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


# =============================================================================
# Saving / reporting
# =============================================================================


def save_graphs(graphs: Sequence, output_dir: Path):
    """Save graphs as torch files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for idx, graph in enumerate(graphs):
        path = output_dir / f"al_fcc_equivariant_alignnd_frame_{idx:04d}.pt"
        torch.save(graph, path)
        paths.append(path)

    return paths


def write_md_rows(rows: Sequence[dict], output_path: Path):
    """Write sampled MD metadata to CSV without requiring pandas."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keys = [
        "step",
        "time_fs",
        "potential_energy_eV",
        "kinetic_energy_eV",
        "total_energy_eV",
        "temperature_K",
    ]

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(str(row[key]) for key in keys) + "\n")


def print_graph_summary(graphs: Sequence, rows: Sequence[dict]):
    """Print a compact graph summary."""
    print("\nEquivariant ALIGNN graph summary")
    print("--------------------------------")
    for idx, (graph, row) in enumerate(zip(graphs, rows)):
        n_nodes = int(graph.num_nodes)
        n_edges = int(graph.edge_index.size(1))
        n_order_edges = int(graph.edge_index_G.size(1)) if getattr(graph, "edge_index_G", None) is not None else -1
        n_angle_edges = (
            int(graph.edge_index_A.size(1))
            if getattr(graph, "edge_index_A", None) is not None
            else 0
        )
        print(
            f"frame={idx:04d} "
            f"step={row['step']:5d} "
            f"time={row['time_fs']:8.2f} fs "
            f"nodes={n_nodes:4d} "
            f"eq_edges={n_edges:5d} "
            f"alignn_edges={n_order_edges:5d} "
            f"angle_edges={n_angle_edges:6d}"
        )


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Al FCC MD, build equivariant ALIGNN graphs, and plot them with NetworkX."
    )

    parser.add_argument("--output-dir", type=Path, default=Path("al_fcc_equivariant_alignnd_example"))
    parser.add_argument("--repeat", type=int, nargs=3, default=(3, 3, 3))
    parser.add_argument("--lattice-constant", type=float, default=4.05)

    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--k", type=int, default=-1)
    parser.add_argument("--include-angs", action="store_true", default=True)
    parser.add_argument("--no-include-angs", dest="include_angs", action="store_false")
    parser.add_argument("--include-dihedrals", action="store_true")
    parser.add_argument("--store-raw-data", action="store_true")

    parser.add_argument("--plot-every-frame", action="store_true")
    parser.add_argument("--projection-axes", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--retry-verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir
    graph_dir = output_dir / "graphs"
    plot_dir = output_dir / "plots"
    traj_path = output_dir / "al_fcc_md.traj"
    metadata_path = output_dir / "md_samples.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    atoms = build_al_fcc_supercell(
        lattice_constant=args.lattice_constant,
        repeat=tuple(args.repeat),
    )

    print(f"Built Al FCC supercell with {len(atoms)} atoms.")
    print(f"Running MD for {args.n_steps} steps at target T={args.temperature_K} K...")

    frames, rows = run_md(
        atoms,
        temperature_K=args.temperature_K,
        timestep_fs=args.timestep_fs,
        n_steps=args.n_steps,
        sample_interval=args.sample_interval,
        seed=args.seed,
        trajectory_path=traj_path,
    )

    write_md_rows(rows, metadata_path)

    print(f"Sampled {len(frames)} frames.")
    print("Building equivariant ALIGNN graphs...")

    graphs = build_equivariant_alignnd_graphs(
        frames,
        cutoff=args.cutoff,
        k=args.k,
        include_angs=args.include_angs,
        include_dihedrals=args.include_dihedrals,
        store_raw_data=args.store_raw_data,
        retry_verbose=args.retry_verbose,
    )

    graph_paths = save_graphs(graphs, graph_dir)
    print_graph_summary(graphs, rows)

    print("\nPlotting NetworkX graphs...")
    if args.plot_every_frame:
        for idx, graph in enumerate(graphs):
            row = rows[idx]
            plot_networkx_graph(
                graph,
                output_path=plot_dir / f"networkx_frame_{idx:04d}.png",
                title=f"Al FCC equivariant ALIGNN graph | step {row['step']}",
                axes=tuple(args.projection_axes),
            )

    plot_graph_montage(
        graphs,
        rows,
        output_path=plot_dir / "networkx_equivariant_graph_montage.png",
        axes=tuple(args.projection_axes),
    )

    plot_md_summary(
        rows,
        output_path=plot_dir / "md_energy_temperature_summary.png",
    )

    # Also save the final frame as a simple XYZ for quick inspection.
    write(output_dir / "al_fcc_final_frame.xyz", frames[-1])

    print("\nWrote outputs:")
    print(f"  trajectory: {traj_path}")
    print(f"  metadata:   {metadata_path}")
    print(f"  graphs:     {graph_dir} ({len(graph_paths)} .pt files)")
    print(f"  plots:      {plot_dir}")
    print("\nMain plot:")
    print(f"  {plot_dir / 'networkx_equivariant_graph_montage.png'}")


if __name__ == "__main__":
    main()
