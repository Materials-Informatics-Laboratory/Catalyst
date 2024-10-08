from ase.neighborlist import neighbor_list

import pandas as pd
import numpy as np

mask2index = lambda mask: np.flatnonzero(mask)

from torch_geometric.data import Data

class AngularGraphPairData(Data):
    """Custom PyG data for representing a pair of two graphs: one for regular atomic
    structure (atom and bonds) and the other for bond/dihedral angles.

    The following arguments assume an atomic graph of `N_atm` atoms with `N_bnd` bonds,
    and an angular graph of `N_ang` angles (including dihedral angles, if there's any).

    Args:
        edge_index_G (LongTensor): Edge index of the atomic graph "G".
        x_atm (Tensor): Atom features.
        x_bnd (Tensor): Bond features.
        edge_index_A (LongTensor): Edge index of the angular graph "A".
        x_ang (Tensor): Angle features.
        mask_dih_ang (Boolean Tensor, optional): If the angular graph contains dihedral
            angles, this mask indicates which angles are dihedral angles.
    """

    def __init__(self,
                 edge_index_G,
                 x_atm,
                 x_bnd,
                 edge_index_A,
                 x_ang,
                 mask_dih_ang=None,
                 ):
        super().__init__()
        self.edge_index_G = edge_index_G
        self.edge_index_A = edge_index_A
        self.x_atm = x_atm
        self.x_bnd = x_bnd
        self.x_ang = x_ang
        self.mask_dih_ang = mask_dih_ang

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_G':
            return self.x_atm.size(0)
        if key == 'edge_index_A':
            return self.x_bnd.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


def index2mask(idx_arr, n):
    mask = np.zeros(n, dtype=int)
    mask[idx_arr] = 1
    return mask.astype(np.bool)

def np_groupby(arr, groups):
    """Numpy implementation of `groupby` operation (a common method in pandas).
    """
    arr, groups = np.array(arr), np.array(groups)
    sort_idx = groups.argsort()
    arr = arr[sort_idx]
    groups = groups[sort_idx]
    return np.split(arr, np.unique(groups, return_index=True)[1])[1:]

def np_scatter(src, index, func):
    """Generalization of the `torch_scatter.scatter` operation for any reduce function.
    See https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html for how `scatter` works.

    Args:
        src (array): The source array.
        index (array of int): The indices of elements to scatter.
        func (function): Reduce function (e.g., mean, sum) that operates on elements with the same indices.

    :rtype: generator
    """
    return (func(g) for g in np_groupby(src, index))

def summary(model):
    """Returns a dataframe describing the numbers of trainable parameters in a torch model.
    """
    params = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    total_num = sum(n for _, n in params)
    params.append(('Total', total_num))
    return pd.DataFrame(params, columns=['Layer', 'Params'])

def atoms2graph(atoms, cutoff):
    """Convert an ASE `Atoms` object into a graph based on a radius cutoff.
    Returns the graph (in COO format) and its edge attributes (format
    determined by `edge_dist`).

    Args:
        atoms (ase.Atoms): Collection of atoms to be converted to a graph.
        cutoff (float): Cutoff radius for nearest neighbor search.
        edge_dist (bool, optional): Set to `True` to output edge distances.
            Otherwise, output edge vectors.

    Returns:
       tuple: Tuple of (edge_index, edge_attr) that describes the atomic graph.

    :rtype: (ndarray, ndarray)
    """
    i, j, d = neighbor_list('ijd', atoms, cutoff)

    return np.stack((i, j)), d.astype(np.float32)

def atoms2knngraph(atoms, cutoff, k=12, scale_inv=True):
    """Convert an ASE `Atoms` object into a graph based on k nearest neighbors.
    Returns the graph (in COO format), and its edge attributes (distance vectors `edge_attr`).

    Args:
        atoms (ase.Atoms): Collection of atoms to be converted to a graph.
        cutoff (float): Cutoff radius for nearest neighbor search.
            These neighbors are then down-selected to k nearest neighbors.
        k (int, optional): Number of nearest neighbors for each atom.
        scale_inv (bool, optional): If set to `True`, normalize the distance
            vectors `edge_attr` such that each atom's furthest neighbor is
            one unit distance away. This makes the knn graph scale-invariant.

    Returns:
       tuple: Tuple of (edge_index, edge_attr) that describes the knn graph.

    :rtype: (ndarray, ndarray)
    """
    edge_src, edge_dst, edge_dists = neighbor_list('ijd', atoms, cutoff=cutoff)

    src_groups  = np_groupby(edge_src, groups=edge_dst)
    dst_groups  = np_groupby(edge_dst, groups=edge_dst)
    dist_groups = np_groupby(edge_dists, groups=edge_dst)

    knn_idx = [np.argsort(d)[:k] for d in dist_groups]
    for indices in knn_idx:
        if len(indices) != k:
            raise Exception("The number of nearest neighbors is not K. Consider increasing the cutoff radius.")

    src_knn = tuple(s[indices] for s, indices in zip(src_groups, knn_idx))
    dst_knn = tuple(d[indices] for d, indices in zip(dst_groups, knn_idx))

    i = np.concatenate(src_knn)
    j = np.concatenate(dst_knn)

    edge_index = np.stack((i, j))
    edge_attr = D.astype(np.float32)

    return edge_index, edge_attr
