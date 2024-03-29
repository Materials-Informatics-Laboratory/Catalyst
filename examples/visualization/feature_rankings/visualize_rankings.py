from pathlib import Path
import shutil
import glob
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import math

def get_top_nperc_rankings(data):
    data = sorted(data, key=float)
    nperc = int(float(len(data))*.1)

    top_data = data[:nperc]
    mean = np.mean(top_data)

    return mean

def plot_data(data):
    from ase.symbols import Symbols

    atom_data = []
    bond_data = []
    angle_data = []

    fig, ax = plt.subplots(nrows=3, ncols=1)

    means = [[],[],[]]
    image = 0
    for at in data["avg_atm_features"][image]:
        atom_data.append(at)
        means[0].append(get_top_nperc_rankings(at))
    for at in data["avg_bnd_features"][image]:
        bond_data.append(at)
        means[1].append(get_top_nperc_rankings(at))
    for at in data["avg_ang_features"][image]:
        angle_data.append(at)
        means[2].append(get_top_nperc_rankings(at))
    all_data = [atom_data,bond_data,angle_data]

    colors = ['pink', 'lightblue', 'lightgreen']
    b1 = ax[0].boxplot(atom_data,patch_artist=True,boxprops=dict(facecolor=colors[0], color='k'))
    b2 = ax[1].boxplot(bond_data,patch_artist=True,boxprops=dict(facecolor=colors[1], color='k'))
    b3 = ax[2].boxplot(angle_data,patch_artist=True,boxprops=dict(facecolor=colors[2], color='k'))

    labels = [item.get_text() for item in ax[0].get_xticklabels()]
    for i,at in enumerate(data["all_atom_types"]):
        labels[i] = at
    ax[0].set_xticklabels(labels)
    labels = [item.get_text() for item in ax[1].get_xticklabels()]
    for i, at in enumerate(data["all_bond_types"]):
        labels[i] = at
    ax[1].set_xticklabels(labels)
    labels = [item.get_text() for item in ax[2].get_xticklabels()]
    for i, at in enumerate(data["all_angle_types"]):
        labels[i] = at
    ax[2].set_xticklabels(labels)

    plt.xticks(rotation=45, ha='right')

    fig2, ax2 = plt.subplots(nrows=1, ncols=3)
    fig3, ax3 = plt.subplots(nrows=1, ncols=3)

    lts = ["all_atom_types", "all_bond_types", "all_angle_types"]
    perc = [1.0, 0.5, 0.1]
    all_labels = []
    all_sub_data = []
    for i, ty in enumerate(means):
        nperc = int(float(len(ty)) * perc[i])
        ty = sorted(enumerate(ty), key=lambda i: i[1])

        bottom_ty = ty[:nperc]
        bottom_index = []
        for b in bottom_ty:
            bottom_index.append(b[0])
        top_ty = ty[-nperc:]
        top_index = []
        for t in top_ty:
            top_index.append(t[0])

        sub_data = [all_data[i][index] for index in bottom_index]
        if i > 0:
            sub_data += [all_data[i][index] for index in top_index]
        sub_labels = [data[lts[i]][index] for index in bottom_index]
        if i > 0:
            sub_labels += [data[lts[i]][index] for index in top_index]

        bp = ax2[i].boxplot(sub_data, patch_artist=True, boxprops=dict(facecolor=colors[i], color='k'))
        vp = ax3[i].violinplot(sub_data)

        violin_parts = vp
        for pc in violin_parts['bodies']:
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')

        labels = [item.get_text() for item in ax2[i].get_xticklabels()]
        for j, at in enumerate(sub_labels):
            labels[j] = at
        ax2[i].set_xticklabels(labels, rotation=45, ha='right')
        ax2[i].set_yscale('log')
        ax3[i].set_yscale('log')
        ax3[i].set_xticklabels(labels, rotation=45, ha='right')

    plt.show()
    plt.show()

def get_total_statistics(all_stats_data, all_data):
    final_data = dict(avg_bnd_features=[],
                        avg_atm_features=[],
                        avg_ang_features=[],
                        std_atm_features=[],
                        std_bnd_features=[],
                        std_ang_features=[],
                        all_atom_types = [],
                        all_bond_types = [],
                        all_angle_types = [])

    all_bonds = []
    all_angles = []
    all_atoms = []

    for data in all_data:
        for bt in data["bnds"]:
            check = 0
            for bond in all_bonds:
                if bond == bt:
                    check = 1
                    break
            if check == 0:
                all_bonds.append(bt)
        for ba in data["angs"]:
            check = 0
            for angle in all_angles:
                if angle == ba:
                    check = 1
                    break
            if check == 0:
                all_angles.append(ba)
        for at in data["atms"]:
            check = 0
            for atom in all_atoms:
                if atom == at:
                    check = 1
                    break
            if check == 0:
                all_atoms.append(at)

    for data in all_stats_data:
        for feature_types in data.items():
            for i, image_vals in enumerate(feature_types[1]):
                final_data[feature_types[0]].append([])
                if 'bnd' in feature_types[0]:
                    for j in range(len(all_bonds)):
                        final_data[feature_types[0]][-1].append([])
                elif 'ang' in feature_types[0]:
                    for j in range(len(all_angles)):
                        final_data[feature_types[0]][-1].append([])
                elif 'atm' in feature_types[0]:
                    for j in range(len(all_atoms)):
                        final_data[feature_types[0]][-1].append([])
        break
    for k,data in enumerate(all_stats_data):
        for feature_types in data.items():
            for i, image_vals in enumerate(feature_types[1]):
                if 'bnd' in feature_types[0]:
                    for n, val in enumerate(image_vals):
                        index = -1
                        for j in range(len(all_bonds)):
                            if all_data[k]["bnds"][n] == all_bonds[j]:
                                index = j
                                break
                        if float(val) > -1.0:
                            final_data[feature_types[0]][i][index].append(float(val))
                if 'ang' in feature_types[0]:
                    for n, val in enumerate(image_vals):
                        index = -1
                        for j in range(len(all_angles)):
                            if all_data[k]["angs"][n] == all_angles[j]:
                                index = j
                                break
                        if float(val) > -1.0:
                            final_data[feature_types[0]][i][index].append(float(val))
                if 'atm' in feature_types[0]:
                    for n, val in enumerate(image_vals):
                        index = -1
                        for j in range(len(all_atoms)):
                            if all_data[k]["atms"][n] == all_atoms[j]:
                                index = j
                                break
                        if float(val) > -1.0:
                            final_data[feature_types[0]][i][index].append(float(val))

    final_data["all_atom_types"] = all_atoms
    final_data["all_bond_types"] = all_bonds
    final_data["all_angle_types"] = all_angles

    return final_data

def get_single_statistics(ind_data):
    avg_data = dict(avg_bnd_features=[],
                    avg_atm_features=[],
                    avg_ang_features=[],
                    std_atm_features=[],
                    std_bnd_features=[],
                    std_ang_features=[])

    for j in range(len(ind_data["atm_amounts"])):
        avg_data["avg_atm_features"].append([])
        avg_data["std_atm_features"].append([])
        for i in ind_data["atms"]:
            avg_data["avg_atm_features"][-1].append([])
            avg_data["std_atm_features"][-1].append([])
        avg_data["avg_bnd_features"].append([])
        avg_data["std_bnd_features"].append([])
        for i in ind_data["bnds"]:
            avg_data["avg_bnd_features"][-1].append([])
            avg_data["std_bnd_features"][-1].append([])
        avg_data["avg_ang_features"].append([])
        avg_data["std_ang_features"].append([])
        for i in ind_data["angs"]:
            avg_data["avg_ang_features"][-1].append([])
            avg_data["std_ang_features"][-1].append([])

    counter = 0
    for j, atm in enumerate(ind_data['i_atm_data']):
        for k in range(len(atm)):
            system_index = -1
            for i, index in enumerate(ind_data["total_indices"][0]):
                if int(ind_data["i_atm_data"][j][k]) < int(index) + (i + 1):
                    system_index = i
                    break
            avg_data["avg_atm_features"][system_index][j].append(float(ind_data["atm_data"][counter]))
            avg_data["std_atm_features"][system_index][j].append(float(ind_data["atm_data"][counter]))
            counter += 1

    for j, bond in enumerate(ind_data["bond_amounts"]):
        for k in range(len(ind_data["i_bnd_data"][j])):
            system_index = -1
            for i, index in enumerate(ind_data["total_indices"][1]):
                if int(ind_data["i_bnd_data"][j][k]) < int(index) + (i + 1):
                    system_index = i
                    break
            avg_data["avg_bnd_features"][system_index][j].append(float(ind_data["bnd_data"][j][k]))
            avg_data["std_bnd_features"][system_index][j].append(float(ind_data["bnd_data"][j][k]))

    for j, angle in enumerate(ind_data["angle_amounts"]):
        for k in range(len(ind_data["i_ang_data"][j])):
            system_index = -1
            for i, index in enumerate(ind_data["total_indices"][2]):
                if int(ind_data["i_ang_data"][j][k]) <= int(index) + (i + 1):
                    system_index = i
                    break
            avg_data["avg_ang_features"][system_index][j].append(float(ind_data["ang_data"][j][k]))
            avg_data["std_ang_features"][system_index][j].append(float(ind_data["ang_data"][j][k]))

    for i,system in enumerate(avg_data["avg_atm_features"]):
        for j,data in enumerate(system):
            if len(data) > 0:
                avg = np.average(data)
                std = np.std(data)

                avg_data["avg_atm_features"][i][j] = avg
                avg_data["std_atm_features"][i][j] = std
            else:
                avg_data["avg_atm_features"][i][j] = -1.0
                avg_data["std_atm_features"][i][j] = -1.0

    for i,system in enumerate(avg_data["avg_bnd_features"]):
        for j,data in enumerate(system):
            if len(data) > 0:
                avg = np.average(data)
                std = np.std(data)

                avg_data["avg_bnd_features"][i][j] = avg
                avg_data["std_bnd_features"][i][j] = std
            else:
                avg_data["avg_bnd_features"][i][j] = -1.0
                avg_data["std_bnd_features"][i][j] = -1.0

    for i,system in enumerate(avg_data["avg_ang_features"]):
        for j,data in enumerate(system):
            if len(data) > 0:
                avg = np.average(data)
                std = np.std(data)

                avg_data["avg_ang_features"][i][j] = avg
                avg_data["std_ang_features"][i][j] = std
            else:
                avg_data["avg_ang_features"][i][j] = -1.0
                avg_data["std_ang_features"][i][j] = -1.0

    return avg_data

path = str(Path(__file__).parent)
results_dir = os.path.join(path,'results')
models = glob.glob(os.path.join(results_dir,'*'))

total_stats_data = []
total_data = []

read_stats = 0
if read_stats:
    final_stats = np.load(os.path.join(path,'ranking_stats.npy'),total_stats_data)
else:
    for model in models:
        files = glob.glob(os.path.join(model,'*'))
        for f in files:
            print('Analyzing ',f.split('\\')[-1])
            rankings_dict = np.load(f, allow_pickle=True)
            rankings_dict = rankings_dict.item()
            amounts_dict = dict(atom_amounts = [],
                                bond_amounts = [],
                                angle_amounts = [],
                                total_indices = [rankings_dict['atm_amounts'],rankings_dict['bnd_amounts'],rankings_dict['ang_amounts']])
            for vals in rankings_dict['i_atm_data']:
                amounts_dict['atom_amounts'].append(len(vals))
            for vals in rankings_dict['i_bnd_data']:
                amounts_dict['bond_amounts'].append(len(vals))
            for vals in rankings_dict['i_ang_data']:
                amounts_dict['angle_amounts'].append(len(vals))
            combined_dict = rankings_dict | amounts_dict

            total_stats_data.append(get_single_statistics(combined_dict))
            total_data.append(combined_dict)
    final_stats = get_total_statistics(total_stats_data,total_data)
    np.save(os.path.join(path, 'ranking_stats.npy'), final_stats)
plot_data(final_stats)




