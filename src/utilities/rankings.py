import numpy as np
import torch as torch
from .structure_properties import get_unique_bonds, get_unique_bond_angles

def organize_rankings(atom_data,bond_data,angle_data,elements,atom_mode='atomic_numbers'):
    bnd_type = []
    ang_type = []
    if atom_mode == 'atomic_numbers':
        output, inverse_indices = torch.unique(atom_data, sorted = False, return_inverse = True)
        output = output[output != 0]
        elements = output
        for i, row in enumerate(atom_data):
            for j, data in enumerate(row):
                if data > 0:
                    atom_data[i][j] = 1
                    break
    else:
        pass
    for i in range(len(bond_data[0])):
        bond1 = atom_data[bond_data[0][i]].numpy()
        bond2 = atom_data[bond_data[1][i]].numpy()
        bond1 = np.where(bond1 == 1)[0][0]
        bond2 = np.where(bond2 == 1)[0][0]

        bond = [elements[bond1],elements[bond2]]
        bnd_type.append(bond)
    #ad0_np = angle_data[0].numpy().tolist()
    #ad1_np = angle_data[1].numpy().tolist()

    ad0_np = angle_data[0].numpy()
    ad1_np = angle_data[1].numpy()
    ang_type = [0]*len(ad0_np)
    for i, a0 in enumerate(ad0_np):
        ang1 = [bond_data[0][a0], bond_data[1][a0]]
        ang2 = [bond_data[0][ad1_np[i]], bond_data[1][ad1_np[i]]]
        bx = [ang1[0].numpy(), ang1[1].numpy(), ang2[0].numpy()]
        for j, b in enumerate(bx):
            bx[j] = atom_data[b].numpy()
        bonds = [np.where(bx[0] == 1)[0][0], np.where(bx[1] == 1)[0][0], np.where(bx[2] == 1)[0][0]]
        angle = [elements[bonds[0]], elements[bonds[1]], elements[bonds[2]]]
        ang_type[i] = angle

    '''
    for ad0,ad1 in zip(ad0_np,ad1_np):
        ang1 = [bond_data[0][ad0], bond_data[1][ad0]]
        ang2 = [bond_data[0][ad1], bond_data[1][ad1]]
        bx = [ang1[0].numpy(), ang1[1].numpy(), ang2[0].numpy()]
        for j,b in enumerate(bx):
            bx[j] = atom_data[b].numpy()
        bonds = [np.where(bx[0] == 1)[0][0], np.where(bx[1] == 1)[0][0], np.where(bx[2] == 1)[0][0]]
        angle = [elements[bonds[0]], elements[bonds[1]], elements[bonds[2]]]
        ang_type.append(angle)
    '''

    '''
    for i in range(len(angle_data[0])):
        ang1 = [bond_data[0][angle_data[0][i]],bond_data[1][angle_data[0][i]]]
        ang2 = [bond_data[0][angle_data[1][i]],bond_data[1][angle_data[1][i]]]
        bx = [ang1[0].numpy(),ang1[1].numpy(),ang2[0].numpy()]
        for j in range(len(bx)):
            bx[j] = atom_data[bx[j]].numpy()
        bonds = [np.where(bx[0] == 1)[0][0],np.where(bx[1] == 1)[0][0],np.where(bx[2] == 1)[0][0]]
        angle = [elements[bonds[0]],elements[bonds[1]],elements[bonds[2]]]
        ang_type.append(angle)
    '''
    i_atm = []
    for i in range(len(elements)):
        i_atm.append([])
        for j,ad in enumerate(atom_data):
            column = np.where(ad.numpy() == 1.0)[0][0]
            if column == i:
                i_atm[-1].append(j)
    x_bnd, i_bnd = get_unique_bonds(bnd_type)
    x_ang, i_ang = get_unique_bond_angles(ang_type)
    return i_atm,x_bnd, i_bnd, x_ang, i_ang
