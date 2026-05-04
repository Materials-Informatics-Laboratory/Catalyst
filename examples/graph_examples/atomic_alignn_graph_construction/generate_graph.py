from pathlib import Path
path = str(Path(__file__).parent)
from catalyst.src.graph.alignnd import alignn_gen
from ase.io import read
import torch as torch
import os
import time

start = time.time()
structures = read(os.path.join(path,'OUTCAR-0'),index=':',format='vasp-out')
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
        'store_atoms_type': 'ase-atoms',
        'cpu_cores':1
    }
    graph_data = alignn_gen(data=data)
    print(graph_data)
    torch.save(graph_data, os.path.join(path, 'graph_data-' + str(i) + '.pt'))
print(f"Time elapsed: {time.time() - start:.4f} seconds")



