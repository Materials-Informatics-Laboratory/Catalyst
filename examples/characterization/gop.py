from catalyst.src.characterization.graph_order_parameter.gop import GOP

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from ase.io import read
import glob as glob
import numpy as np
import os

path = str(Path(__file__).parent)
data_file = os.path.join(path,'OUTCAR')

params = dict(
    cutoffs=[3.0,4.0,5.0], # [3.0,4.0,5.0,...]
    interactions=[['W','W']], # [['Al','Al'],...]
    k=3
)
snapshots = read(data_file,index=':')
gop = GOP(params=params)
preds, feature_vectors = gop.predict(snapshots,flatten=True)
pca = PCA(n_components=2)
x = pca.fit_transform(feature_vectors)

plt.scatter(x[:,0],x[:,1])
plt.show()


