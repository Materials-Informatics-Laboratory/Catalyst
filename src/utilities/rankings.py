import numpy as np
import torch as torch
from src.properties.structure_properties import get_unique_2body, get_unique_3body
from src.properties.physics_database import Physics_data

def organize_rankings_atomic(data,atom_data,bond_data,angle_data,atom_mode='atomic_numbers'):

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
        pdb = Physics_data()
        ohe = []
        elements = []
        for el in pdb.elements:
            elements.append(el.number)
    for i in range(len(bond_data[0])):
        bond1 = atom_data[bond_data[0][i]].numpy()
        bond2 = atom_data[bond_data[1][i]].numpy()
        bond1 = np.where(bond1 == 1)[0][0]
        bond2 = np.where(bond2 == 1)[0][0]

        bond = [elements[bond1],elements[bond2]]
        bnd_type.append(bond)
    if hasattr(data,'x_ang'):
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
        x_ang, i_ang = get_unique_3body(ang_type)
    else:
        x_ang = None
        i_ang = None

    i_atm = []
    for i in range(len(elements)):
        i_atm.append([])
        for j,ad in enumerate(atom_data):
            column = np.where(ad.numpy() == 1.0)[0][0]
            if column == i:
                i_atm[-1].append(j)
    x_bnd, i_bnd = get_unique_2body(bnd_type)

    return i_atm,x_bnd, i_bnd, x_ang, i_ang, elements

def organize_rankings_generic(data,g_node_data,a_node_data,a_edge_data):

    ng_type = [ii for ii in range(len(g_node_data[0]))]
    na_type = []
    ea_type = []

    for i in range(len(a_node_data[0])):
        a1 = g_node_data[a_node_data[0][i]].numpy()
        a2 = g_node_data[a_node_data[1][i]].numpy()
        a1 = np.where(a1 == 1)[0][0]
        a2 = np.where(a2 == 1)[0][0]

        na = [ng_type[a1],ng_type[a2]]
        na_type.append(na)
    if hasattr(data,'edge_A') or hasattr(data,'x_ang'):
        ad0_np = a_edge_data[0].numpy()
        ad1_np = a_edge_data[1].numpy()
        ea_type = [0]*len(ad0_np)
        for i, a0 in enumerate(ad0_np):
            aa1 = [a_node_data[0][a0], a_node_data[1][a0]]
            aa2 = [a_node_data[0][ad1_np[i]], a_node_data[1][ad1_np[i]]]
            bx = [aa1[0].numpy(), aa1[1].numpy(), aa2[0].numpy()]
            for j, b in enumerate(bx):
                bx[j] = g_node_data[b].numpy()
            two_body = [np.where(bx[0] == 1)[0][0], np.where(bx[1] == 1)[0][0], np.where(bx[2] == 1)[0][0]]
            three_body = [ng_type[two_body[0]], ng_type[two_body[1]], ng_type[two_body[2]]]
            ea_type[i] = three_body
        x_ea, i_ea = get_unique_3body(ea_type)
    else:
        x_ea = None
        i_ea = None

    i_ng = []
    for i in range(len(ng_type)):
        i_ng.append([])
        for j,ad in enumerate(g_node_data):
            column = np.where(ad.numpy() == 1.0)[0][0]
            if column == i:
                i_ng[-1].append(j)
    x_na, i_na = get_unique_2body(na_type)

    return i_ng,x_na, i_na, x_ea, i_ea, ng_type