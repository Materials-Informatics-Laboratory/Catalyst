from pathlib import Path
import shutil
import glob
import os

from ase.io import read
import torch as torch

from catalyst.src.utilities.stoichiometry import get_structure_stoichiometry, check_stoichiometry, check_num_elements
from catalyst.src.physics.properties import calc_reaction_enthalpy
from catalyst.src.graph.alignnd import realignnd

# PREPROCESS DATA
path = str(Path(__file__).parent)
graph_dir = os.path.join(path,'data','graphs')
if os.path.isdir(graph_dir):
    shutil.rmtree(graph_dir)
os.mkdir(graph_dir)

dataset_A = 'A'
dataset_B = 'B'
dataset_C = 'C'
pathways = [[dataset_A,dataset_B,dataset_C]]

A_files = glob.glob(os.path.join(path,'data',dataset_A,'*'))
B_files = glob.glob(os.path.join(path,'data',dataset_B,'*'))
C_files = glob.glob(os.path.join(path,'data',dataset_C,'*'))

energy = [[],[],[]] # A,B,C
stoichiometries = [[],[],[]]
snapshots = [[],[],[]]
for i,structure in enumerate(A_files):
    outcars = glob.glob(os.path.join(structure, 'OUTCAR*'))
    snapshots[0].append(read(outcars[-1], format='vasp-out', index='-1:'))
for i,structure in enumerate(B_files):
    outcars = glob.glob(os.path.join(structure, 'OUTCAR*'))
    snapshots[1].append(read(outcars[-1], format='vasp-out', index='-1:'))
for i,structure in enumerate(C_files):
    outcars = glob.glob(os.path.join(structure, 'OUTCAR*'))
    snapshots[2].append(read(outcars[-1], format='vasp-out', index='-1:'))

y_scale = 10.0
for i,A_structure in enumerate(A_files):
    stoichiometries[0].append(get_structure_stoichiometry(snapshots[0][i][-1]))
    scale = float(len(snapshots[0][i][-1]))
    energy[0].append(snapshots[0][i][-1].get_potential_energy() / scale)
    for j, B_structure in enumerate(B_files):
        stoichiometries[1].append(get_structure_stoichiometry(snapshots[1][j][-1]))
        scale = float(len(snapshots[1][j][-1]))
        energy[1].append(snapshots[1][j][-1].get_potential_energy() / scale)
        for k, C_structure in enumerate(C_files):
            stoichiometries[2].append(get_structure_stoichiometry(snapshots[2][k][-1]))
            scale = float(len(snapshots[2][k][-1]))
            energy[2].append(snapshots[2][k][-1].get_potential_energy() / scale)
            if check_num_elements(stoichiometries[0][-1],[stoichiometries[1][-1],stoichiometries[2][-1]]):
                x = check_stoichiometry(stoichiometries[0][-1],[stoichiometries[1][-1],stoichiometries[2][-1]],delta=0.15)
                if x:
                    H_mix = calc_reaction_enthalpy(energies=[energy[0][-1],energy[1][-1],energy[2][-1]],n_systems=len(energy))
                    structures = [snapshots[0][i][-1],snapshots[1][j][-1],snapshots[2][k][-1]]
                    graph_data = realignnd(structures, cutoff=4.0, dihedral=False,store_atoms=False,use_pt=True)
                    graph_data.y = torch.tensor(H_mix + y_scale)
                    torch.save(graph_data, os.path.join(graph_dir, graph_data.gid + '.pt'))
