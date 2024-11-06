from catalyst.src.characterization.graph_order_parameter.gop import GOP
from catalyst.src.graph.alignnd import alignnd

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read
import glob as glob
import numpy as np
import os

path = str(Path(__file__).parent)
data_file = os.path.join(path,'OUTCAR-0')

params = dict(
    cutoffs=[2.0,3.0,4.0
             ], # [3.0,4.0,5.0,...]
    interactions=[[[1,0,0,0,0],[1,0,0,0,0]],[[1,0,0,0,0],[0,0,0,0,1]]], # [['Al','Al'],...]
    k=3
)
fig, ax = plt.subplots(nrows=1, ncols=2,sharex=True,sharey=False)
snapshots = read(data_file,index=':')
'''
Calculate VGOP using Graph data
'''
dataset = []
for snapshot in snapshots:
    graph_data = alignnd(snapshot, cutoff=6.0, dihedral=False, store_atoms=False, use_pt=False)
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
    cutoffs=[3.0,4.0,5.0,6.0], # [3.0,4.0,5.0,...]
    interactions=[['Co','Co']], # [['Al','Al'],...]
    k=3
)
gop = GOP(params=params)
preds, feature_vectors = gop.predict(snapshots,flatten=True)
pca = PCA(n_components=2)
x = pca.fit_transform(feature_vectors)
ax[1].scatter(x[:,0],x[:,1])
plt.show()


