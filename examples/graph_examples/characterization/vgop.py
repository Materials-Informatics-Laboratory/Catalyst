from catalyst.src.characterization.graph_order_parameter.gop import GOP
from catalyst.src.graph.alignnd import alignn_gen

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read
import glob as glob
import numpy as np
import random
import os


from ase import Atoms
from ase.io import read
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units

def create_alcu_slab(size=(4, 4, 5), vacuum=10.0, concentration=0.5):
    """Creates an AlCu(111) slab with a given concentration of Cu."""

    slab = fcc111('Al', size=size, vacuum=vacuum)
    slab.center(axis=2)

    # Randomly replace Al atoms with Cu to achieve the desired concentration
    num_al_atoms = len(slab)
    num_cu_atoms = int(num_al_atoms * concentration)

    indices_to_replace = random.sample(range(num_al_atoms), num_cu_atoms)

    for index in indices_to_replace:
        slab[index].symbol = 'Cu'

    return slab


path = str(Path(__file__).parent)
slab = create_alcu_slab(concentration=0.5) # 50% Cu, 50% Al.
slab.calc = EMT()
temperature = 500 * units.kB  # 500 K in eV
MaxwellBoltzmannDistribution(slab, temperature * units.kB) # in eV
dyn = VelocityVerlet(slab, 2 * units.fs)  # 2 fs timestep
dyn = Langevin(slab, 5 * units.fs, temperature_K=10000, friction=0.02) # friction is in units of 1/ps
traj = Trajectory(os.path.join(path,'md.traj'), 'w', slab)
dyn.attach(traj, interval=10) # save every 10 steps.
dyn.run(500) # run again to save the trajectory

fig, ax = plt.subplots(nrows=1, ncols=2)
snapshots = read(os.path.join(path,'md.traj'),index=':')
'''
Calculate VGOP using Graph data
'''

params = dict(
    cutoffs=[3.0,4.0,5.0
             ], # [3.0,4.0,5.0,...]
    interactions=[[[1,0],[0,1]],[[0,1],[0,1]],[[1,0],[1,0]]], # [['Al','Cu'],...]
    k=3
)
dataset = []
for snapshot in snapshots:
    data = {
        'type': 'alignnd',
        'neighbor_params': [5.0, -1],
        'raw_data': snapshot,
        'is_dihedral': False,
        'include_angs': False,
    }
    graph_data = alignn_gen(data=data)
    dataset.append(graph_data)
gop = GOP(params=params)
preds, feature_vectors = gop.predict(dataset,flatten=True)
pca = PCA(n_components=2)
x = pca.fit_transform(feature_vectors)
ax[0].scatter(x[:,0],x[:,1])

'''
Calculate VGOP using ASE Atoms
'''

params = dict(
    cutoffs=[3.0,4.0,5.0], # [3.0,4.0,5.0,...]
    interactions=[['Al','Cu'],['Cu','Cu'],['Al','Al']], # [['Al','Cu'],...]
    k=3
)
gop = GOP(params=params)
preds, feature_vectors = gop.predict(snapshots,flatten=True)
pca = PCA(n_components=2)
x = pca.fit_transform(feature_vectors)
ax[1].scatter(x[:,0],x[:,1])

plt.show()


