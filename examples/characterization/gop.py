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
    cutoffs=[3.0,4.0,5.0,6.0], # [3.0,4.0,5.0,...]
    interactions=[['W','W']], # [['Al','Al'],...]
    k=3
)
snapshots = read(data_file,index=':')
gop = GOP(params=params)
preds = gop.predict(snapshots)
feature_vectors = []
for pred in preds:
    feature_vectors.append([])
    for p1 in pred:
        for p2 in p1:
            if len(feature_vectors[-1]) == 0:
                feature_vectors[-1] = p2
            else:
                feature_vectors[-1] = feature_vectors[-1] + p2
pca = PCA(n_components=2)
x = pca.fit_transform(feature_vectors)

plt.scatter(x[:,0],x[:,1])
plt.show()


