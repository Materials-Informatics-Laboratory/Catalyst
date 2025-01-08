from pathlib import Path
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from catalyst.src.properties.physics_database import Physics_data
from catalyst.src.io.io import load_dictionary

def get_top_nperc_rankings(data):
    data = sorted(data, key=float)
    nperc = int(float(len(data))*.25)

    top_data = data[:nperc]
    mean = np.mean(top_data)

    return mean

def plot_data(data):
    atom_data = []
    bond_data = []
    angle_data = []

    fig, ax = plt.subplots(nrows=3, ncols=1)

    means = [[],[],[]]
    image = 0
    for at in data["avg_ng_features"][image]:
        atom_data.append(at)
        means[0].append(get_top_nperc_rankings(at))
    for at in data["avg_na_features"][image]:
        bond_data.append(at)
        means[1].append(get_top_nperc_rankings(at))
    for at in data["avg_ea_features"][image]:
        angle_data.append(at)
        means[2].append(get_top_nperc_rankings(at))
    all_data = [atom_data,bond_data,angle_data]

    colors = ['pink', 'lightblue', 'lightgreen']
    b1 = ax[0].boxplot(atom_data,patch_artist=True,boxprops=dict(facecolor=colors[0], color='k'))
    b2 = ax[1].boxplot(bond_data,patch_artist=True,boxprops=dict(facecolor=colors[1], color='k'))
    b3 = ax[2].boxplot(angle_data,patch_artist=True,boxprops=dict(facecolor=colors[2], color='k'))

    fixed_labels = []
    labels = [item.get_text() for item in ax[0].get_xticklabels()]

    for i,at in enumerate(data["all_ng_types"]):
        labels[i] = at
    fixed_labels.append(labels)
    from ase import Atoms
    elements = labels.copy()
    fake_system = Atoms(elements, positions=[(0,0,0)]*len(elements))
    numbers = fake_system.get_atomic_numbers()

    ax[0].set_xticklabels(labels)
    labels = [item.get_text() for item in ax[1].get_xticklabels()]
    for i, at in enumerate(data["all_na_types"]):
        labels[i] = at

    fixed_labels.append(labels)
    ax[1].set_xticklabels(labels)

    labels = [item.get_text() for item in ax[2].get_xticklabels()]
    for i, at in enumerate(data["all_ea_types"]):
        labels[i] = at

    fixed_labels.append(labels)
    ax[2].set_xticklabels(labels)

    plt.xticks(rotation=45, ha='right')

    fig2, ax2 = plt.subplots(nrows=1, ncols=3)
    fig3, ax3 = plt.subplots(nrows=1, ncols=3)

    lts = ["all_ng_types", "all_na_types", "all_ea_types"]
    perc = [1.0, 0.25, 0.25]
    all_labels = []
    all_sub_data = []
    data = fixed_labels
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
        sub_labels = [data[i][index] for index in bottom_index]
        if i > 0:
            sub_labels += [data[i][index] for index in top_index]

        bp = ax2[i].boxplot(sub_data, patch_artist=True, boxprops=dict(facecolor=colors[i], color='k'))

        sub_data = [x for x in sub_data if x != []]
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
    final_data = dict(avg_na_features=[],
                        avg_ng_features=[],
                        avg_ea_features=[],
                        std_ng_features=[],
                        std_na_features=[],
                        std_ea_features=[],
                        all_ng_types = [],
                        all_na_types = [],
                        all_ea_types = [])

    all_na = []
    all_ea = []
    all_ng = []

    for data in all_data:
        for bt in data["na"]:
            check = 0
            for bond in all_na:
                if bond == bt:
                    check = 1
                    break
            if check == 0:
                all_na.append(bt)
        if data["ea"] is not None:
            for ba in data["ea"]:
                check = 0
                for angle in all_ea:
                    if angle == ba:
                        check = 1
                        break
                if check == 0:
                    all_ea.append(ba)
        for at in data["ng"]:
            check = 0
            for atom in all_ng:
                if atom == at:
                    check = 1
                    break
            if check == 0:
                all_ng.append(at)

    for data in all_stats_data:
        for feature_types in data.items():
            for i, image_vals in enumerate(feature_types[1]):
                final_data[feature_types[0]].append([])
                if 'avg_na' in feature_types[0]:
                    for j in range(len(all_na)):
                        final_data[feature_types[0]][-1].append([])
                elif 'avg_ea' in feature_types[0]:
                    for j in range(len(all_ea)):
                        final_data[feature_types[0]][-1].append([])
                elif 'avg_ng' in feature_types[0]:
                    for j in range(len(all_ng)):
                        final_data[feature_types[0]][-1].append([])
        break
    for k,data in enumerate(all_stats_data):
        for feature_types in data.items():
            for i, image_vals in enumerate(feature_types[1]):
                if 'avg_na' in feature_types[0]:
                    for n, val in enumerate(image_vals):
                        index = -1
                        for j in range(len(all_na)):
                            if all_data[k]["na"][n] == all_na[j]:
                                index = j
                                break
                        if float(val) > -1.0:
                            final_data[feature_types[0]][i][index].append(float(val))
                if 'avg_ea' in feature_types[0]:
                    for n, val in enumerate(image_vals):
                        index = -1
                        for j in range(len(all_ea)):
                            if all_data[k]["ea"][n] == all_ea[j]:
                                index = j
                                break
                        if float(val) > -1.0:
                            final_data[feature_types[0]][i][index].append(float(val))
                if 'avg_ng' in feature_types[0]:
                    for n, val in enumerate(image_vals):
                        index = -1
                        for j in range(len(all_ng)):
                            if all_data[k]["ng"][n] == all_ng[j]:
                                index = j
                                break
                        if float(val) > -1.0:
                            final_data[feature_types[0]][i][index].append(float(val))

    final_data["all_ng_types"] = all_ng
    final_data["all_na_types"] = all_na
    final_data["all_ea_types"] = all_ea

    return final_data

def get_single_statistics(ind_data,take=0):
    avg_data = dict(avg_na_features=[],
                    avg_ng_features=[],
                    avg_ea_features=[],
                    std_ng_features=[],
                    std_na_features=[],
                    std_ea_features=[])

    avg_data["avg_ng_features"].append([])
    avg_data["std_ng_features"].append([])
    pdb = Physics_data()
    elems = []
    for i,e in enumerate(ind_data["ng"]):
        for el in pdb.elements:
            if el.symbol == e:
                ind_data["ng"][i] = el.number
                break
    for i in ind_data["ng"]:
        avg_data["avg_ng_features"][-1].append([])
        avg_data["std_ng_features"][-1].append([])
    avg_data["avg_na_features"].append([])
    avg_data["std_na_features"].append([])
    for i in ind_data["na"]:
        avg_data["avg_na_features"][-1].append([])
        avg_data["std_na_features"][-1].append([])
    avg_data["avg_ea_features"].append([])
    avg_data["std_ea_features"].append([])
    if ind_data["ea"] is not None:
        for i in ind_data["ea"]:
            avg_data["avg_ea_features"][-1].append([])
            avg_data["std_ea_features"][-1].append([])

    counter = 0
    for j, atm in enumerate(ind_data['i_ng_data']):
        for k in range(len(atm)):
            system_index = -1
            for i, index in enumerate(ind_data["total_indices"][0]):
                if int(ind_data["i_ng_data"][j][k]) < int(index) + (i + 1):
                    system_index = i
                    break
            if system_index == take:
                el_counter = np.where(np.array(ind_data["ng"]) == j)
                avg_data["avg_ng_features"][0][el_counter[0][0]].append(float(ind_data["ng_data"][counter]))
                avg_data["std_ng_features"][0][el_counter[0][0]].append(float(ind_data["ng_data"][counter]))
            counter += 1
    bnd_counts = {v: np.where(ind_data["i_na_data"] == v)[0] for v in np.unique(ind_data["i_na_data"])}
    if ind_data["total_indices"][1].size()[0] == 0:
        ind_data["total_indices"][1] = torch.tensor([10000000000])
    for j, bond_type in enumerate(ind_data["na_amounts"]):  # bond type
        if take == 0:
            indices = np.where(bnd_counts[j] < ind_data["total_indices"][1][take].item())
        else:
            indices = np.where((bnd_counts[j] < ind_data["total_indices"][1][take - 1].item()) & (
                        bnd_counts[j] < ind_data["total_indices"][1][take].item()))
        for index in indices[0]:
            avg_data["avg_na_features"][0][j].append(float(ind_data["na_data"][bnd_counts[j][index]]))
            avg_data["std_na_features"][0][j].append(float(ind_data["na_data"][bnd_counts[j][index]]))
    if ind_data["ea"] is not None:
        ang_counts = {v: np.where(ind_data["i_ea_data"] == v)[0] for v in np.unique(ind_data["i_ea_data"])}
        for j, ang_type in enumerate(ind_data["ea_amounts"]):  # bond type
            if take == 0:
                indices = np.where(ang_counts[j] < ind_data["total_indices"][2][take].item())
            else:
                indices = np.where((ang_counts[j] < ind_data["total_indices"][2][take - 1].item()) &
                                   (ang_counts[j] < ind_data["total_indices"][2][take].item()))
            for index in indices[0]:
                avg_data["avg_ea_features"][0][j].append(float(ind_data["ea_data"][ang_counts[j][index]]))
                avg_data["std_ea_features"][0][j].append(float(ind_data["ea_data"][ang_counts[j][index]]))
    for i,system in enumerate(avg_data["avg_ng_features"]):
        for j,data in enumerate(system):
            if len(data) > 0:
                avg = np.average(data)
                std = np.std(data)

                avg_data["avg_ng_features"][i][j] = avg
                avg_data["std_ng_features"][i][j] = std
            else:
                avg_data["avg_ng_features"][i][j] = -1.0
                avg_data["std_ng_features"][i][j] = -1.0

    for i,system in enumerate(avg_data["avg_na_features"]):
        for j,data in enumerate(system):
            if len(data) > 0:
                avg = np.average(data)
                std = np.std(data)

                avg_data["avg_na_features"][i][j] = avg
                avg_data["std_na_features"][i][j] = std
            else:
                avg_data["avg_na_features"][i][j] = -1.0
                avg_data["std_na_features"][i][j] = -1.0
    if ind_data["ea"] is not None:
        for i,system in enumerate(avg_data["avg_ea_features"]):
            for j,data in enumerate(system):
                if len(data) > 0:
                    avg = np.average(data)
                    std = np.std(data)

                    avg_data["avg_ea_features"][i][j] = avg
                    avg_data["std_ea_features"][i][j] = std
                else:
                    avg_data["avg_ea_features"][i][j] = -1.0
                    avg_data["std_ea_features"][i][j] = -1.0

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
            #rankings_dict = np.load(f, allow_pickle=True)
            rankings_dict = load_dictionary(f)
            #print(rankings_dict)
            #rankings_dict = rankings_dict.item()
            amounts_dict = dict(ng_amounts = [],
                                na_amounts = [],
                                ea_amounts = [],
                                total_indices = [rankings_dict['ng_amounts'],rankings_dict['na_amounts'],rankings_dict['ea_amounts']])
            for vals in rankings_dict['i_ng_data']:
                amounts_dict['ng_amounts'].append(len(vals))
            values, counts = np.unique(rankings_dict['i_na_data'], return_counts=True)
            for count in counts:
                amounts_dict['na_amounts'].append(count)
            values, counts = np.unique(rankings_dict['i_ea_data'], return_counts=True)
            for count in counts:
                amounts_dict['ea_amounts'].append(count)
            combined_dict = rankings_dict | amounts_dict

            total_stats_data.append(get_single_statistics(combined_dict,take=0))
            total_data.append(combined_dict)
    final_stats = get_total_statistics(total_stats_data,total_data)
    np.save(os.path.join(path, 'ranking_stats.npy'), final_stats)
plot_data(final_stats)




