from ase.neighborlist import neighbor_list
from scipy.spatial import cKDTree
from functools import partial
import pandas as pd
import numpy as np
import itertools
import secrets
import sys

from torch_geometric.data import Data

import numba
from numba import njit, prange

mask2index = lambda mask: np.flatnonzero(mask)

class Generic_Graph_Data(Data):
    """Custom PyG data for representing a pair of two graphs: one for the original graph and the other for its line grpah variant.
    This data is used to represent an arbitrary graph and does not hard-code variables based on their atomistic connection.
    """

    def __init__(self,
                 edge_index_G,
                 node_G,
                 reference=None,
                 edge_index_A=None,
                 node_A=None,
                 edge_A=None,
                 mask_edge_A=None,
                 node_G_amounts = None,
                 node_A_amounts = None,
                 edge_A_amounts = None
                 ):
        super().__init__()
        self.reference = reference
        self.edge_index_G = edge_index_G
        self.edge_index_A = edge_index_A
        self.node_G = node_G
        self.node_A = node_A
        self.edge_A = edge_A
        self.node_G_amounts = node_G_amounts
        self.node_A_amounts = node_A_amounts
        self.edge_A_amounts = edge_A_amounts
        self.gid = None

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_G':
            return self.node_G.size(0)
        if key == 'edge_index_A':
            return self.node_A.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def generate_gid(self):
        self.gid = secrets.token_hex(64)

class Atomic_Graph_Data(Data):
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
                 atoms,
                 edge_index_G,
                 edge_index_A,
                 x_atm,
                 x_bnd,
                 x_ang,
                 atm_amounts,
                 bnd_amounts,
                 ang_amounts,
                 mask_dih_ang=None
                 ):
        super().__init__()
        self.atoms = atoms
        self.edge_index_G = edge_index_G
        self.edge_index_A = edge_index_A
        self.x_atm = x_atm
        self.x_bnd = x_bnd
        self.x_ang = x_ang
        self.mask_dih_ang = mask_dih_ang
        self.atm_amounts = atm_amounts
        self.bnd_amounts = bnd_amounts
        self.ang_amounts = ang_amounts

        self.gid = None

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_G':
            return self.x_atm.size(0)
        if key == 'edge_index_A':
            return self.x_bnd.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def generate_gid(self):
        self.gid = secrets.token_hex(64)

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

import numpy as np
import numba
from numba import njit, prange

@njit
def minimum_image(dx, box_length):
    if dx > 0.5 * box_length:
        dx -= box_length
    elif dx < -0.5 * box_length:
        dx += box_length
    return dx

@njit
def build_cell_list(positions, box, cell_size):
    n_atoms = positions.shape[0]
    nx = int(box[0] // cell_size)
    ny = int(box[1] // cell_size)
    nz = int(box[2] // cell_size)
    if nx == 0: nx = 1
    if ny == 0: ny = 1
    if nz == 0: nz = 1

    n_cells = nx * ny * nz
    atom_cell_indices = np.empty(n_atoms, dtype=np.int32)

    for i in range(n_atoms):
        cx = int(positions[i,0] / cell_size) % nx
        cy = int(positions[i,1] / cell_size) % ny
        cz = int(positions[i,2] / cell_size) % nz
        atom_cell_indices[i] = cx + cy*nx + cz*nx*ny

    counts = np.zeros(n_cells, dtype=np.int32)
    for c in atom_cell_indices:
        counts[c] += 1

    cell_starts = np.zeros(n_cells, dtype=np.int32)
    for i in range(1, n_cells):
        cell_starts[i] = cell_starts[i-1] + counts[i-1]

    cell_ends = np.zeros(n_cells, dtype=np.int32)
    for i in range(n_cells):
        cell_ends[i] = cell_starts[i] + counts[i]

    cell_list_atoms = np.empty(n_atoms, dtype=np.int32)
    counters = np.zeros(n_cells, dtype=np.int32)
    for i in range(n_atoms):
        c = atom_cell_indices[i]
        pos = cell_starts[c] + counters[c]
        cell_list_atoms[pos] = i
        counters[c] += 1

    return nx, ny, nz, cell_starts, cell_ends, cell_list_atoms, atom_cell_indices

@njit
def get_neighbor_cells(cx, cy, cz, nx, ny, nz):
    neighbors = []
    for dx in (-1,0,1):
        nxn = (cx + dx) % nx
        for dy in (-1,0,1):
            nyn = (cy + dy) % ny
            for dz in (-1,0,1):
                nzn = (cz + dz) % nz
                neighbors.append(nxn + nyn*nx + nzn*nx*ny)
    return neighbors

@njit(parallel=True)
def find_neighbors_cell_list(positions, box, cell_size, cutoff):
    n_atoms = positions.shape[0]
    cutoff_sq = cutoff * cutoff
    nx, ny, nz, cell_starts, cell_ends, cell_list_atoms, atom_cell_indices = build_cell_list(positions, box, cell_size)

    max_edges_per_thread = 10000  # Adjust based on memory, e.g. ~max expected neighbors per thread
    n_threads = numba.get_num_threads()

    # create buffers per thread:
    edge_i_threads = np.empty((n_threads, max_edges_per_thread), dtype=np.int32)
    edge_j_threads = np.empty((n_threads, max_edges_per_thread), dtype=np.int32)
    edge_d_threads = np.empty((n_threads, max_edges_per_thread), dtype=np.float32)
    edge_count_threads = np.zeros(n_threads, dtype=np.int32)

    for atom_idx in prange(n_atoms):
        thread_id = numba.get_thread_id()
        pos_i = positions[atom_idx]

        cx = int(pos_i[0] / cell_size) % nx
        cy = int(pos_i[1] / cell_size) % ny
        cz = int(pos_i[2] / cell_size) % nz

        neighbor_cells = get_neighbor_cells(cx, cy, cz, nx, ny, nz)

        for cell in neighbor_cells:
            start = cell_starts[cell]
            end = cell_ends[cell]
            for idx in range(start, end):
                j = cell_list_atoms[idx]
                if j <= atom_idx:
                    continue

                dx = positions[j,0] - pos_i[0]
                dy = positions[j,1] - pos_i[1]
                dz = positions[j,2] - pos_i[2]

                if dx > 0.5 * box[0]: dx -= box[0]
                elif dx < -0.5 * box[0]: dx += box[0]
                if dy > 0.5 * box[1]: dy -= box[1]
                elif dy < -0.5 * box[1]: dy += box[1]
                if dz > 0.5 * box[2]: dz -= box[2]
                elif dz < -0.5 * box[2]: dz += box[2]

                r2 = dx*dx + dy*dy + dz*dz
                if r2 <= cutoff_sq:
                    count = edge_count_threads[thread_id]
                    if count == max_edges_per_thread:
                        # Could handle buffer extension or skip overflow edges safely
                        continue
                    edge_i_threads[thread_id, count] = atom_idx
                    edge_j_threads[thread_id, count] = j
                    edge_d_threads[thread_id, count] = np.sqrt(r2)
                    edge_count_threads[thread_id] = count + 1

    # After parallel: gather all thread edges together
    total_edges = np.sum(edge_count_threads)
    edge_i = np.empty(total_edges, dtype=np.int32)
    edge_j = np.empty(total_edges, dtype=np.int32)
    edge_d = np.empty(total_edges, dtype=np.float32)

    pos = 0
    for t in range(n_threads):
        count = edge_count_threads[t]
        edge_i[pos:pos+count] = edge_i_threads[t, :count]
        edge_j[pos:pos+count] = edge_j_threads[t, :count]
        edge_d[pos:pos+count] = edge_d_threads[t, :count]
        pos += count

    return edge_i, edge_j, edge_d

'''
def atoms2graph(atoms, cutoff, k=5):
    """
    Fast neighbor finder for atoms based on cell linked list with PBC.
    Fully replaces ASE neighbor_list and KDTree approaches.

    Returns:
        edge_index (2 x N_edges ndarray)
        edge_attr distances as ndarray (N_edges,)
    """
    atoms.wrap()  # wrap positions into the box
    pos = atoms.get_positions()
    box = np.linalg.norm(atoms.cell[:3], axis=1)
    cell_size = cutoff

    edge_i, edge_j, edge_d = find_neighbors_cell_list(pos, box, cell_size, cutoff)

    return np.stack((edge_i, edge_j)), edge_d.astype(np.float32)

'''
def atoms2graph(atoms, cutoff,k=5):
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
    if k < 0:
        i, j, d = neighbor_list('ijd', atoms, cutoff)
    else:
        atoms.wrap()
        lv_norm = [np.linalg.norm(atoms.cell[0]).item(),
                   np.linalg.norm(atoms.cell[1]).item(),
                   np.linalg.norm(atoms.cell[2]).item()]

        tree = cKDTree(data=atoms.get_positions(),boxsize=lv_norm)
        dd, ii = tree.query(x=atoms.get_positions(),k=k+1,distance_upper_bound=2.0*cutoff)

        i = [[m] * k for m in range(len(ii))]
        i = [n for one_dim in i for n in one_dim]
        j = [sublist[1:] for sublist in ii]
        j = [n for one_dim in j for n in one_dim]
        d = [sublist[1:] for sublist in dd]
        d = [n for one_dim in d for n in one_dim]

    return np.stack((i, j)), np.array(d).astype(np.float32)



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

permute_2 = partial(itertools.permutations, r=2)
def line_graph(edge_index_G):
    """Return the (angular) line graph of the input graph.

    Args:
        edge_index_G (ndarray): Input graph in COO format.
    """
    src_G, dst_G = edge_index_G
    edge_index_A = [
        (u, v)
        for edge_pairs in np_scatter(np.arange(len(dst_G)), dst_G, permute_2)
        for u, v in edge_pairs
    ]
    return np.array(edge_index_A).T

def dihedral_graph(edge_index_G):
    """Return the "dihedral angle line graph" of the input graph.

    Args:
        edge_index_G (ndarray): Input graph in COO format.
    """
    src, dst = edge_index_G
    edge_index_A = [
        (u, v)
        for i, j in edge_index_G.T
        for u in np.flatnonzero((dst == i) & (src != j))
        for v in np.flatnonzero((dst == j) & (src != i))
    ]
    return np.array(edge_index_A).T

'''
DEPRECIATED CLASS: To remove prior to v1.0
'''
class Graph_Data(Data):
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
                 atoms,
                 edge_index_G,
                 edge_index_A,
                 x_atm,
                 x_bnd,
                 x_ang,
                 atm_amounts,
                 bnd_amounts,
                 ang_amounts,
                 mask_dih_ang=None
                 ):
        super().__init__()
        self.atoms = atoms
        self.edge_index_G = edge_index_G
        self.edge_index_A = edge_index_A
        self.x_atm = x_atm
        self.x_bnd = x_bnd
        self.x_ang = x_ang
        self.mask_dih_ang = mask_dih_ang
        self.atm_amounts = atm_amounts
        self.bnd_amounts = bnd_amounts
        self.ang_amounts = ang_amounts

        self.gid = None

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_G':
            return self.x_atm.size(0)
        if key == 'edge_index_A':
            return self.x_bnd.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def generate_gid(self):
        self.gid = secrets.token_hex(64)
