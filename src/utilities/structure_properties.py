import numpy as np
import torch

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
def get_3body_angle(data,edge_index_G, edge_index_A):
    """Return the bond angles (in radians) for the (angular) line graph edges.
    """
    indices = edge_index_G.T[edge_index_A.T].reshape(-1, 4)[:,[0,1,2]]
    angles = []
    for indice in indices:
        v1 = data[indice[0]] - data[indice[1]]
        v2 = data[indice[2]] - data[indice[1]]
        angles.append(angle_between(v1, v2))
    return np.radians(angles)

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

def get_unique_bonds(bonds):
    sums = []
    for bnd in bonds:
        if torch.is_tensor(bnd):
            bnd = bnd.tolist()
        sums.append(sum(bnd))
    unique_bonds, indices = np.unique(np.array(sums), return_inverse=True)

    '''
    unique_bonds = []
    indicies = []

    counter = 0
    while counter < len(bonds):
        if len(unique_bonds) == 0:
            unique_bonds.append(bonds[counter])
            indicies.append([])
        else:
            check = 0
            for i, ub in enumerate(unique_bonds):
                if (bonds[counter][0] == ub[0] and bonds[counter][1] == ub[1]) or (bonds[counter][0] == ub[1] and bonds[counter][1] == ub[0]):
                    check = 1
                    break
            if check == 0:
                unique_bonds.append(bonds[counter])
                indicies.append([])
        counter += 1
    for i,bond in enumerate(bonds):
        for j,ub in enumerate(unique_bonds):
            if (bond[0] == ub[0] and bond[1] == ub[1]) or (bond[0] == ub[1] and bond[1] == ub[0]):
                indicies[j].append(i)
    '''
    return unique_bonds, indices

def get_unique_bond_angles(angles):

    sums = []
    for ang in angles:
        if torch.is_tensor(ang):
            ang = ang.tolist()
        sums.append(sum(ang))
    unique_angles , indices = np.unique(np.array(sums), return_inverse=True)

    '''
    counter = 0
    check = 0
    while counter < len(angles):
        if len(unique_angles) == 0:
            unique_angles.append(angles[counter])
            indicies.append([])
        else:
            for i, ua in enumerate(unique_angles):
                check = 0
                if (angles[counter][0] == ua[0] and angles[counter][1] == ua[1] and angles[counter][2] == ua[2]) or (
                        angles[counter][0] == ua[2] and angles[counter][1] == ua[0] and angles[counter][2] == ua[0]):
                    check = 1
                    break
            if check == 0:
                unique_angles.append(angles[counter])
                indicies.append([])
        counter += 1
    for i, angle in enumerate(angles):
        for j, ua in enumerate(unique_angles):
            if (angle[0] == ua[0] and angle[1] == ua[1] and angle[2] == ua[2]) or (
                    angle[0] == ua[2] and angle[1] == ua[0] and angle[2] == ua[0]):
                indicies[j].append(i)
                break
    '''
    return unique_angles, indices
