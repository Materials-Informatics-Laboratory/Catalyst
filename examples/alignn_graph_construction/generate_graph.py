from pathlib import Path
path = str(Path(__file__).parent)
from catalyst.src.graph.alignnd import alignnd
from ase.io import read
import torch as torch

structures = read(os.path.join(path,'OUTCAR_Al_FCC'),index=':',format='vasp-out')
dataset = []
for i,structure in enumerate(structures):
    print('Generating graph for structure ',i,' of ',len(structures))
    graph_data = alignnd(structure,cutoff=6.0,dihedral=False,store_atoms=False,use_pt=False)
    dataset.append(graph_data)
torch.save(dataset, os.path.join(path, 'graph_data.pt'))



