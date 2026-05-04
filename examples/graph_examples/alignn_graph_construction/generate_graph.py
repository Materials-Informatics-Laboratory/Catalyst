from pathlib import Path
path = str(Path(__file__).parent)
from catalyst.src.graph.alignnd import alignn_gen
import matplotlib.pyplot as plt
from ase.io import read
import torch as torch
import networkx as nx
import numpy as np
import os

structures = read(os.path.join(path,'OUTCAR-0'),index=':',format='vasp-out')
dataset = []
for i,structure in enumerate(structures):
    print('Generating graph for structure ',i,' of ',len(structures))
    data = {
        'type': 'alignnd',
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
    dataset.append(graph_data)
torch.save(dataset, os.path.join(path, 'graph_data.pt'))



