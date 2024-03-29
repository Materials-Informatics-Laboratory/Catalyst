from pathlib import Path
path = str(Path(__file__).parent)
from catalyst.src.graph.alignnd import atomic_alignnd
from ase.io import read
import torch as torch
import os

structures = read(os.path.join(path,'OUTCAR_Al_FCC'),index=':',format='vasp-out')
dataset = []
for i,structure in enumerate(structures):
    print('Generating graph for structure ',i,' of ',len(structures))
    atomic_graphs = atomic_alignnd(structure,cutoff=6.0,dihedral=False)
    torch.save(atomic_graphs, os.path.join(path, 'graph_data.pt'))




