import numpy as np

def get_bnd_angs(atoms, edge_index_G, edge_index_A_bnd_ang):
    """Return the bond angles (in radians) for the (angular) line graph edges.
    """
    indices = edge_index_G.T[edge_index_A_bnd_ang.T].reshape(-1, 4)
    bnd_angs = atoms.get_angles(indices[:, [0, 1, 2]])
    return np.radians(bnd_angs)

def get_dih_angs(atoms, edge_index_G, edge_index_A_dih_ang):
    """Return the dihedral angles (in radians) for the dihedral line graph edges.
    """
    indices = edge_index_G.T[edge_index_A_dih_ang.T].reshape(-1, 4)
    dih_angs = atoms.get_dihedrals(indices[:, [0, 1, 3, 2]])
    return np.radians(dih_angs)
