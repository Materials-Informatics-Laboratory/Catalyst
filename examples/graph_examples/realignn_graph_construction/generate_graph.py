from pathlib import Path
path = str(Path(__file__).parent)
from catalyst.src.graph.alignnd import alignn_gen
import matplotlib.pyplot as plt
from ase.io import read
import torch as torch
import networkx as nx
import numpy as np
import os

def visualize_graph(data, atomic=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    colors = np.array([
        'aqua', 'mediumslateblue', 'peru', 'limegreen',
        'darkorange', 'salmon', 'brown', 'gold'
    ])

    edge_index_G = data.edge_index_G.detach().cpu().numpy()
    x_atm = data.x_atm.detach().cpu().numpy()

    # Use real atomic positions if atoms are stored.
    if getattr(data, "atoms", None) is not None:
        pos = data.atoms.get_positions()[:, :2]
    else:
        # Fast fallback layout: points on a circle.
        n_atoms = x_atm.shape[0]
        theta = np.linspace(0, 2 * np.pi, n_atoms, endpoint=False)
        pos = np.column_stack([np.cos(theta), np.sin(theta)])

    color_ids = np.argmax(x_atm == 1.0, axis=1)
    color_map = colors[color_ids % len(colors)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # ------------------------
    # Graph G: atomic graph
    # ------------------------
    segments_G = pos[edge_index_G.T]

    ax[0].add_collection(
        LineCollection(
            segments_G,
            colors='dimgrey',
            linewidths=0.4,
            zorder=1
        )
    )

    ax[0].scatter(
        pos[:, 0],
        pos[:, 1],
        s=100,
        c=color_map,
        edgecolors='black',
        linewidths=0.4,
        zorder=2
    )

    ax[0].set_title('Graph G (1,2 body graph)')
    ax[0].axis('equal')
    ax[0].axis('off')

    # ------------------------
    # Graph A: angular graph
    # Nodes are bonds from G.
    # Positions are bond midpoints.
    # ------------------------
    if data.edge_index_A is not None:
        edge_index_A = data.edge_index_A.detach().cpu().numpy()

        bond_pos = 0.5 * (
            pos[edge_index_G[0]] + pos[edge_index_G[1]]
        )

        segments_A = bond_pos[edge_index_A.T]

        ax[1].add_collection(
            LineCollection(
                segments_A,
                colors='dimgrey',
                linewidths=0.4,
                zorder=1
            )
        )

        ax[1].scatter(
            bond_pos[:, 0],
            bond_pos[:, 1],
            s=50,
            c='lightgrey',
            edgecolors='black',
            linewidths=0.3,
            zorder=2
        )

        ax[1].set_title('Graph A (2,3 body graph)')
        ax[1].axis('equal')
        ax[1].axis('off')
    else:
        ax[1].set_title('Graph A not included')
        ax[1].axis('off')

    plt.tight_layout()
    plt.show()
print('Reading data and generating graph...')
structures = [read(os.path.join(path,'OUTCAR-0'),index=':-1',format='vasp-out')[0],
    read(os.path.join(path,'OUTCAR-1'),index=':-1',format='vasp-out')[0],
    read(os.path.join(path,'OUTCAR-2'),index=':-1',format='vasp-out')[0]
              ]
data = {
        'type': 'realignnd',
        'neighbor_params': [3.0, -1],
        'raw_data': structures,
        'is_dihedral': False,
        'store_raw_data': False,
        'use_pt': False,
        'include_angs': True,
        'node_labels': None,
        'element_list': ['Al'],
        'store_atoms_type': 'ase-atoms'
    }
graph_data = alignn_gen(data=data)
print(graph_data)
torch.save(graph_data, os.path.join(path, 'graph_data.pt'))
visualize_graph(graph_data)



