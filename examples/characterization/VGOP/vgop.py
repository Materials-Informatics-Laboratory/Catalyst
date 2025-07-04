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

def nvt_melting_md(atoms, temperature_range, steps, timestep, friction, output_file):
    """
    Performs NVT molecular dynamics to simulate melting of a bulk material.

    Args:
        symbol (str): Chemical symbol of the material (e.g., 'Al', 'Cu').
        temperature_range (tuple): Tuple of (initial_temperature, final_temperature) in Kelvin.
        steps (int): Total number of MD steps.
        timestep (float): MD timestep in femtoseconds.
        friction (float): Langevin thermostat friction parameter (1/ps).
        output_file (str): Name of the trajectory file to save.
    """
    atoms.calc = EMT()
    dyn = Langevin(atoms, timestep * units.fs, temperature_K=temperature_range[0], friction=friction)
    traj = Trajectory(output_file, 'w', atoms)
    dyn.attach(traj, interval=10)
    initial_temperature, final_temperature = temperature_range
    temperature_step = (final_temperature - initial_temperature) / steps
    for step in range(steps):
        current_temperature = initial_temperature + temperature_step * step
        dyn.temperature_K = current_temperature # adjust the thermostat temperature.
        dyn.run(1)
        print(f"Step: {step+1}, Temperature: {current_temperature:.2f} K")

    print("NVT melting simulation complete.")

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
nvt_melting_md(atoms=slab,temperature_range=(100,1500), steps=1000, timestep=5, friction=0.02,output_file=os.path.join(path,'md.traj'))

fig, ax = plt.subplots(nrows=1, ncols=2)
snapshots = read(os.path.join(path,'md.traj'),index=':')
'''
Calculate VGOP using Graph data
'''

params = dict(
    cutoffs=[2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0], # [3.0,4.0,5.0,...]
    interactions=[[[1,0],[0,1]],[[0,1],[0,1]],[[1,0],[1,0]]], # [['Al','Cu'],...]
    k=3
)
dataset = []
for snapshot in snapshots:
    data = {
        'type': 'alignnd',
        'neighbor_params': [6.0, -1],
        'raw_data': snapshot,
        'is_dihedral': False,
        'include_angs': False,
    }
    graph_data = alignn_gen(data=data)
    dataset.append(graph_data)
gop = GOP(params=params)
preds, feature_vectors = gop.predict(dataset,flatten=True)
pca = PCA(n_components=8)
x = pca.fit_transform(feature_vectors)
c1 = np.linspace(0,len(snapshots),len(snapshots))
scatter1 = ax[0].scatter(x[:,0],x[:,1],c=c1, cmap='viridis')
cbar1 = fig.colorbar(scatter1, ax=ax[0])
'''
Calculate VGOP using ASE Atoms
'''

gop = GOP(params=params)
preds, feature_vectors = gop.predict(snapshots,flatten=True)
pca = PCA(n_components=8)
x = pca.fit_transform(feature_vectors)
c2 = np.linspace(0,len(snapshots),len(snapshots))
scatter2 = ax[1].scatter(x[:,0],x[:,1],c=c2, cmap='viridis')
cbar2 = fig.colorbar(scatter2, ax=ax[1])

plt.show()


