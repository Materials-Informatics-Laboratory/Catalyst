from pathlib import Path
path = str(Path(__file__).parent)
from catalyst.src.graph.alignnd import alignn_gen
import matplotlib.pyplot as plt
from ase.io import read
import torch as torch
import networkx as nx
import numpy as np
import os

def visualize_graph(data,atomic=False):
    # Drawing options
    G_options = {
        'edgecolors': 'black',
        'width': 0.4,
        'font_size': 16,
        'node_size': 100,
    }

    edge_index_bnd = data.edge_index_G.numpy()
    G = nx.Graph(list(edge_index_bnd.T))
    G_pos = nx.spring_layout(G)
    color_map = []
    colors = ['aqua','mediumslateblue','peru','limegreen','darkorange','salmon','brown','gold']
    for node in data.x_atm:
        x = np.where(node == 1.0)[0][0]
        color_map.append(colors[x])
    fig, ax = plt.subplots(1, 2)
    nx.draw_networkx(G, G_pos, **G_options, with_labels=False, node_color=color_map, edge_color='dimgrey',
                         arrows=False, ax=ax[0])

    edge_index_A = data.edge_index_A.numpy()
    A = nx.Graph(list(edge_index_A.T))
    A_pos = nx.spring_layout(A)
    nx.draw_networkx(A, A_pos, **G_options, with_labels=False, edge_color='dimgrey',
                     arrows=False, ax=ax[1])
    ax[0].set_title('Graph G (1,2 body graph)')
    ax[1].set_title('Graph A (2,3 body graph)')

    plt.draw()
    plt.show()

structures = read(os.path.join(path,'OUTCAR_Al_FCC'),index=':',format='vasp-out')
dataset = []
for i,structure in enumerate(structures):
    print('Generating graph for structure ',i,' of ',len(structures))
    data = {
        'type': 'atomic_alignnd',
        'neighbor_params': [3.0, -1],
        'raw_data': structure,
        'is_dihedral': False,
        'store_raw_data': False,
        'use_pt': False,
        'include_angs': True,
        'node_labels': None,
        'element_list': ['Al'],
        'store_atoms_type': 'ase-atoms'
    }
    graph_data = alignn_gen(data=data)
    torch.save(graph_data, os.path.join(path, 'graph_data-' + str(i) + '.pt'))




