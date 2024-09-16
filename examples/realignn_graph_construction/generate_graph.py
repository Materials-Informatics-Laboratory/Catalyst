from catalyst.src.graph.alignnd import realignnd
from pathlib import Path
from ase.io import read
import torch as torch
import sys
import os

path = str(Path(__file__).parent)
structures = [read(os.path.join(path,'OUTCAR-0'),index=':-1',format='vasp-out')[0],
    read(os.path.join(path,'OUTCAR-1'),index=':-1',format='vasp-out')[0],
    read(os.path.join(path,'OUTCAR-2'),index=':-1',format='vasp-out')[0]
              ]
graph_data = realignnd(structures,cutoff=5.0,dihedral=False,store_atoms=False,use_pt=True)
torch.save(graph_data, os.path.join(path, 'graph_data.pt'))



