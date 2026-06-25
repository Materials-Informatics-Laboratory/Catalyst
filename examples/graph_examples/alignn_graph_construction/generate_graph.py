from pathlib import Path
import os

import torch
from ase.io import read

from catalyst.graph.alignnd import alignn_gen


path = Path(__file__).parent

structures = read(
    path / "OUTCAR-0",
    index=":",
    format="vasp-out",
)

dataset = []

for i, structure in enumerate(structures):
    print(f"Generating graph for structure {i + 1} of {len(structures)}")

    data = {
        "type": "alignnd",
        "raw_data": structure,

        # [cutoff, k]
        # k = -1 means cutoff-based graph generation in your atoms2graph setup.
        "neighbor_params": [3.0, -1],

        "is_dihedral": False,
        "store_raw_data": False,
        "use_pt": False,
        "include_angs": True,
        "node_labels": None,
        "element_list": ["Al"],
        "store_atoms_type": "ase-atoms",

        # New retry / robustness controls
        "auto_retry_graph": True,
        "max_graph_attempts": 8,
        "k_step": 4,
        "cutoff_scale": 1.25,
        "max_k": None,
        "max_cutoff": 6.0,
        "require_bonds": True,
        "require_angles": True,
        "require_dihedrals": False,
        "retry_verbose": True,

        # Used if the updated alignnd supports chunked angle/dihedral calls.
        # Safe to leave as 1 if not using internal angle parallelism.
        "cpu_cores": 1,
    }

    graph_data = alignn_gen(data=data)
    dataset.append(graph_data)

torch.save(dataset, path / "graph_data.pt")