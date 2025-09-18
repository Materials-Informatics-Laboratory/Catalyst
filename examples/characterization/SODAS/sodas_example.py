from catalyst.src.ml.nn.gnn.models.alignn import Encoder_generic,Encoder_atomic, Processor, Decoder,PositiveScalarsDecoder, ALIGNN
from catalyst.src.characterization.sodas.model.sodas import SODAS
from catalyst.src.graph.generic_build import generic_graph_gen
import catalyst.src.utilities.sampling as sampling
from catalyst.src.io.data_management import load_dictionary, save_dictionary
from catalyst.src.observer.params import Catalyst
from catalyst.src.characterization.sodas.utils.utils import generate_latent_space_path, assign_gammas
from catalyst.src.utilities.data_tools import parallel_sort

from torch_geometric.loader import DataLoader
import torch as torch
from torch import nn

from pathlib import Path, PurePath
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import shutil
import glob
import os

from sklearn.neighbors import KDTree
from umap import umap_
import networkx as nx
import random
import math

'''
Global parameter initialization
'''
global n_types
global projection_indim
global projection_outdim
global cutoff
global n_convs
global n_data
global n_nodes
global n_dim

'''
Function definitions
'''
def visualize_graph(data,atomic=False):
    # Drawing options
    G_options = {
        'edgecolors': 'black',
        'width': 0.4,
        'font_size': 16,
        'node_size': 100,
    }

    edge_index_bnd = data.edge_index_G.numpy()
    G = nx.Graph(list(edge_index_bnd.T))
    G_pos = nx.spring_layout(G)
    color_map = []
    colors = ['aqua','mediumslateblue','peru','limegreen','darkorange','salmon','brown','gold']
    for node in data.node_G:
        x = np.where(node == 1.0)[0][0]
        color_map.append(colors[x])
    fig, ax = plt.subplots(1, 2)
    nx.draw_networkx(G, G_pos, **G_options, with_labels=False, node_color=color_map, edge_color='dimgrey',
                         arrows=False, ax=ax[0])

    edge_index_A = data.edge_index_A.numpy()
    A = nx.Graph(list(edge_index_A.T))
    A_pos = nx.spring_layout(A)
    nx.draw_networkx(A, A_pos, **G_options, with_labels=False, edge_color='dimgrey',
                     arrows=False, ax=ax[1])
    ax[0].set_title('Graph G (1,2 body graph)')
    ax[1].set_title('Graph A (2,3 body graph)')

    plt.draw()
    plt.show()

def generate_data(cat,visualize_final=False):
    '''
    DATA INITIALIZATION AND GRAPH CONSTRUCTION
    '''
    if os.path.isdir(cat.parameters['io_dict']['data_dir']):
        shutil.rmtree(cat.parameters['io_dict']['data_dir'])
    os.mkdir(cat.parameters['io_dict']['data_dir'])

    k = np.linspace(3,9,n_data) # number of neighbors per graph node
    dataset = []
    y = np.linspace(0,1,n_data)
    for ds in range(n_data):
        if ds % 500 == 0:
            print('Generating graph ',ds)
        data = np.random.uniform(-1,1, size=(math.ceil(n_nodes[ds]), n_dim))  # randomly create raw data
        g_node_labels = np.eye(n_types)[np.random.choice(n_types, len(data))]  # randomly assign G node labels
        tree = KDTree(data,metric='euclidean',leaf_size=2)
        dist, ind = tree.query(data,k=math.ceil(k[ds])+1) # k+1 due to self-interaction, neighbor list
        neighbor_data = {
            'dist':dist,
            'ind':ind,
            'g_nodes':g_node_labels
        }
        graph_gen_data = {
            'raw_data':data,
            'params':neighbor_data,
            'line_graph':True,
            'type':'generic_pairwise'
        }
        graph = generic_graph_gen(graph_gen_data)
        graph.y = torch.tensor(y[ds],dtype=torch.float)
        torch.save(graph, os.path.join(os.path.join(cat.parameters['io_dict']['main_path'],cat.parameters['io_dict']['data_dir']), graph.gid + '.pt'))
    if visualize_final:
        visualize_graph(graph, atomic=False)

def project_data(cat):
    '''
    PROJECT DATA
    '''
    graph_data = [torch.load(file_name) for file_name in
                      glob.glob(os.path.join(cat.parameters['io_dict']['data_dir'], '*'))]
    # read data and perform projections
    print('Performing graph projections...')
    cat.parameters['io_dict']['projection_dir'] = os.path.join(cat.parameters['io_dict']['main_path'],'projections')
    if os.path.isdir(cat.parameters['io_dict']['projection_dir']):
        shutil.rmtree(cat.parameters['io_dict']['projection_dir'])
    os.mkdir(cat.parameters['io_dict']['projection_dir'])
    cat.parameters['io_dict']['samples_dir'] = os.path.join(cat.parameters['io_dict']['main_path'], 'samples')
    if os.path.isdir(cat.parameters['io_dict']['samples_dir']):
        shutil.rmtree(cat.parameters['io_dict']['samples_dir'])
    os.mkdir(cat.parameters['io_dict']['samples_dir'])
    projected_data = None
    encoded_data = []
    gids = []
    y = []
    follow_batch = ['node_G', 'node_A', 'edge_A'] if hasattr(graph_data[0], 'edge_A') else ['node_G', 'node_A']
    for data in graph_data:
        gids.append(data.gid)
        y.append(data.y)
    loader = DataLoader(graph_data, batch_size=parameters['loader_dict']['batch_size'], shuffle=False, follow_batch=follow_batch,
                                num_workers=cat.parameters['loader_dict']['num_workers'])
    encoded_data = cat.parameters['model_dict']['model'].generate_gnn_latent_space(parameters=cat.parameters,loader=loader)

    encoded_data = np.array(encoded_data)
    cat.parameters['model_dict']['model'].fit_preprocess(data=encoded_data)
    cat.parameters['model_dict']['model'].fit_dim_red(data=encoded_data)
    projected_data = cat.parameters['model_dict']['model'].project_data(data=encoded_data)
    stored_projections = dict(
            projections=projected_data,
            gids=gids,
            y = y,
            graphs= graph_data
    )
    #save_dictionary(os.path.join(cat.parameters['io_dict']['projection_dir'], 'projection_data.npy'), stored_projections)
    return graph_data, stored_projections

def determine_global_gammas(ml,global_data):
    '''
    order projections in terms of y value
    '''
    import matplotlib.colors as mcolors
    y_copy = [y.item() for y in global_data['y']]
    pp,yy = parallel_sort(global_data['projections'],y_copy)
    global_data['projections'] = np.array(pp)
    global_data['y'] = yy

    fig, ax = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)

    path_data = generate_latent_space_path(global_data['projections'], boundaries=[75,-75], k=4,version='1',random_neighbors=False)
    assigned_gammas = assign_gammas(global_data['projections'], global_data['projections'], path_data, smearing='sum',k=1,
                                    iterations=1, cutoff=10.0
                                    )
    ax[0][0].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=global_data['y'])

    ax[0][1].plot(path_data['weighted_path'][:, 0],
                  path_data['weighted_path'][:, 1], color='k', markeredgecolor='r', marker='o',
                  linestyle='-', markerfacecolor='w')
    ax[0][1].scatter(path_data['weighted_path'][0][0],
                     path_data['weighted_path'][0][1], c='b', s=250)
    ax[0][1].scatter(path_data['weighted_path'][-1][0],
                     path_data['weighted_path'][-1][1], c='k', s=250)

    ax[0][2].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=assigned_gammas)

    deltas = []
    for i in range(len(global_data['graphs'])):
        delta = global_data['y'][i] - assigned_gammas[i]
        deltas.append(delta)
    norm = mcolors.Normalize(vmin=min(deltas), vmax=max(deltas))
    mappable = cm.ScalarMappable(norm=norm, cmap='terrain')
    fig.colorbar(mappable, ax=ax[0][3])

    ax[0][3].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=deltas,
                     cmap='terrain', edgecolor='k')
    ax[0][3].set_title(np.average(np.array(deltas)))

    assigned_gammas = assign_gammas(global_data['projections'], global_data['projections'], path_data, smearing='tanh',
                                    k=1,
                                    iterations=4, cutoff=10.0
                                    )
    #fig, ax = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)
    ax[1][0].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=global_data['y'])

    ax[1][1].plot(path_data['weighted_path'][:, 0],
                  path_data['weighted_path'][:, 1], color='k', markeredgecolor='r', marker='o',
                  linestyle='-', markerfacecolor='w')
    ax[1][1].scatter(path_data['weighted_path'][0][0],
                     path_data['weighted_path'][0][1], c='b', s=250)
    ax[1][1].scatter(path_data['weighted_path'][-1][0],
                     path_data['weighted_path'][-1][1], c='k', s=250)

    ax[1][2].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=assigned_gammas)

    deltas = []
    for i in range(len(global_data['graphs'])):
        delta = global_data['y'][i] - assigned_gammas[i]
        deltas.append(delta)
    norm = mcolors.Normalize(vmin=min(deltas), vmax=max(deltas))
    mappable = cm.ScalarMappable(norm=norm, cmap='terrain')
    fig.colorbar(mappable, ax=ax[1][3])

    ax[1][3].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=deltas,
                     cmap='terrain', edgecolor='k')
    ax[1][3].set_title(np.average(np.array(deltas)))

    path_data = generate_latent_space_path(global_data['projections'], boundaries=[75,-75], k=4,version='1',random_neighbors=True)

    assigned_gammas = assign_gammas(global_data['projections'], global_data['projections'], path_data, smearing='sum',
                                    k=1,
                                    iterations=1, cutoff=10.0
                                    )
    #fig, ax = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)
    ax[2][0].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=global_data['y'])

    ax[2][1].plot(path_data['weighted_path'][:, 0],
                  path_data['weighted_path'][:, 1], color='k', markeredgecolor='r', marker='o',
                  linestyle='-', markerfacecolor='w')
    ax[2][1].scatter(path_data['weighted_path'][0][0],
                     path_data['weighted_path'][0][1], c='b', s=250)
    ax[2][1].scatter(path_data['weighted_path'][-1][0],
                     path_data['weighted_path'][-1][1], c='k', s=250)

    ax[2][2].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=assigned_gammas)

    deltas = []
    for i in range(len(global_data['graphs'])):
        delta = global_data['y'][i] - assigned_gammas[i]
        deltas.append(delta)
    norm = mcolors.Normalize(vmin=min(deltas), vmax=max(deltas))
    mappable = cm.ScalarMappable(norm=norm, cmap='terrain')
    fig.colorbar(mappable, ax=ax[2][3])

    ax[2][3].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=deltas,
                     cmap='terrain', edgecolor='k')
    ax[2][3].set_title(np.average(np.array(deltas)))

    assigned_gammas = assign_gammas(global_data['projections'], global_data['projections'], path_data,smearing='tanh',k=10,
                                    iterations=4,cutoff=10.0,scale=10.0
                                    )

    ax[3][0].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=global_data['y'])

    ax[3][1].plot(path_data['weighted_path'][:, 0],
                  path_data['weighted_path'][:, 1], color='k', markeredgecolor='r', marker='o',
                  linestyle='-', markerfacecolor='w')
    ax[3][1].scatter(path_data['weighted_path'][0][0],
                     path_data['weighted_path'][0][1], c='b', s=250)
    ax[3][1].scatter(path_data['weighted_path'][-1][0],
                     path_data['weighted_path'][-1][1], c='k', s=250)

    ax[3][2].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=assigned_gammas)

    deltas = []
    for i in range(len(global_data['graphs'])):
        delta = global_data['y'][i] - assigned_gammas[i]
        deltas.append(delta)
    norm = mcolors.Normalize(vmin=min(deltas), vmax=max(deltas))
    mappable = cm.ScalarMappable(norm=norm, cmap='terrain')
    fig.colorbar(mappable, ax=ax[3][3])

    ax[3][3].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=deltas,
                     cmap='terrain', edgecolor='k')
    ax[3][3].set_title(np.average(np.array(deltas)))

    fig2, ax2 = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)

    path_data = generate_latent_space_path(global_data['projections'], boundaries=[75, -75], k=4, version='2',
                                           random_neighbors=False,reduction=0.05)
    assigned_gammas = assign_gammas(global_data['projections'], global_data['projections'], path_data, smearing='sum',
                                    k=1,
                                    iterations=1, cutoff=10.0
                                    )
    ax2[0][0].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=global_data['y'])

    ax2[0][1].plot(path_data['weighted_path'][:, 0],
                  path_data['weighted_path'][:, 1], color='k', markeredgecolor='r', marker='o',
                  linestyle='-', markerfacecolor='w')
    ax2[0][1].scatter(path_data['weighted_path'][0][0],
                     path_data['weighted_path'][0][1], c='b', s=250)
    ax2[0][1].scatter(path_data['weighted_path'][-1][0],
                     path_data['weighted_path'][-1][1], c='k', s=250)

    ax2[0][2].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=assigned_gammas)

    deltas = []
    for i in range(len(global_data['graphs'])):
        delta = global_data['y'][i] - assigned_gammas[i]
        deltas.append(delta)
    norm = mcolors.Normalize(vmin=min(deltas), vmax=max(deltas))
    mappable = cm.ScalarMappable(norm=norm, cmap='terrain')
    fig2.colorbar(mappable, ax=ax2[0][3])

    ax2[0][3].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=deltas,
                     cmap='terrain', edgecolor='k')
    ax2[0][3].set_title(np.average(np.array(deltas)))

    assigned_gammas = assign_gammas(global_data['projections'], global_data['projections'], path_data, smearing='tanh',
                                    k=1,
                                    iterations=4, cutoff=10.0
                                    )
    # fig, ax = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)
    ax2[1][0].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=global_data['y'])

    ax2[1][1].plot(path_data['weighted_path'][:, 0],
                  path_data['weighted_path'][:, 1], color='k', markeredgecolor='r', marker='o',
                  linestyle='-', markerfacecolor='w')
    ax2[1][1].scatter(path_data['weighted_path'][0][0],
                     path_data['weighted_path'][0][1], c='b', s=250)
    ax2[1][1].scatter(path_data['weighted_path'][-1][0],
                     path_data['weighted_path'][-1][1], c='k', s=250)

    ax2[1][2].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=assigned_gammas)

    deltas = []
    for i in range(len(global_data['graphs'])):
        delta = global_data['y'][i] - assigned_gammas[i]
        deltas.append(delta)
    norm = mcolors.Normalize(vmin=min(deltas), vmax=max(deltas))
    mappable = cm.ScalarMappable(norm=norm, cmap='terrain')
    fig2.colorbar(mappable, ax=ax2[1][3])

    ax2[1][3].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=deltas,
                     cmap='terrain', edgecolor='k')
    ax2[1][3].set_title(np.average(np.array(deltas)))

    path_data = generate_latent_space_path(global_data['projections'], boundaries=[75, -75], k=4, version='2',
                                           random_neighbors=True,reduction=0.005)

    assigned_gammas = assign_gammas(global_data['projections'], global_data['projections'], path_data, smearing='sum',
                                    k=1,
                                    iterations=1, cutoff=10.0
                                    )
    ax2[2][0].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=global_data['y'])

    ax2[2][1].plot(path_data['weighted_path'][:, 0],
                  path_data['weighted_path'][:, 1], color='k', markeredgecolor='r', marker='o',
                  linestyle='-', markerfacecolor='w')
    ax2[2][1].scatter(path_data['weighted_path'][0][0],
                     path_data['weighted_path'][0][1], c='b', s=250)
    ax2[2][1].scatter(path_data['weighted_path'][-1][0],
                     path_data['weighted_path'][-1][1], c='k', s=250)

    ax2[2][2].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=assigned_gammas)

    deltas = []
    for i in range(len(global_data['graphs'])):
        delta = global_data['y'][i] - assigned_gammas[i]
        deltas.append(delta)
    norm = mcolors.Normalize(vmin=min(deltas), vmax=max(deltas))
    mappable = cm.ScalarMappable(norm=norm, cmap='terrain')
    fig2.colorbar(mappable, ax=ax2[2][3])

    ax2[2][3].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=deltas,
                     cmap='terrain', edgecolor='k')
    ax2[2][3].set_title(np.average(np.array(deltas)))

    # path_data = generate_latent_space_path(global_data['projections'], boundaries=[50,-50], k=4,version='2',reduction=0.01,random_neighbors=True)
    assigned_gammas = assign_gammas(global_data['projections'], global_data['projections'], path_data, smearing='tanh',
                                    k=10,
                                    iterations=10, cutoff=10.0, scale=10.0
                                    )

    ax2[3][0].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=global_data['y'])

    ax2[3][1].plot(path_data['weighted_path'][:, 0],
                  path_data['weighted_path'][:, 1], color='k', markeredgecolor='r', marker='o',
                  linestyle='-', markerfacecolor='w')
    ax2[3][1].scatter(path_data['weighted_path'][0][0],
                     path_data['weighted_path'][0][1], c='b', s=250)
    ax2[3][1].scatter(path_data['weighted_path'][-1][0],
                     path_data['weighted_path'][-1][1], c='k', s=250)

    ax2[3][2].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=assigned_gammas)

    deltas = []
    for i in range(len(global_data['graphs'])):
        delta = global_data['y'][i] - assigned_gammas[i]
        deltas.append(delta)
    norm = mcolors.Normalize(vmin=min(deltas), vmax=max(deltas))
    mappable = cm.ScalarMappable(norm=norm, cmap='terrain')
    fig2.colorbar(mappable, ax=ax2[3][3])

    ax2[3][3].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=deltas,
                     cmap='terrain', edgecolor='k')
    ax2[3][3].set_title(np.average(np.array(deltas)))

    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[0][2])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[0][0])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[1][2])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[1][0])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[2][2])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[2][0])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[3][2])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[3][0])

    fig2.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax2[0][2])
    fig2.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax2[0][0])
    fig2.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax2[1][2])
    fig2.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax2[1][0])
    fig2.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax2[2][2])
    fig2.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax2[2][0])
    fig2.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax2[3][2])
    fig2.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax2[3][0])

    plt.show()

if __name__ == '__main__':
    n_types = 2  # number of ficticious types to label each node in G
    projection_indim = 100
    projection_outdim = 100
    cutoff = 10.0
    n_convs = 3
    n_data = 1000 # total number of samples
    n_nodes = np.linspace(5, 50, n_data)  # number of data points per sample
    n_dim = 4  # number of dimensions in intial raw data
    parameters = dict(
        device_dict=dict(
            world_size=1,
            device='cpu',
            ddp_backend='gloo',
            run_ddp=False,
            pin_memory=False,
            find_unused_parameters=False
        ),
        io_dict=dict(
            main_path=str(Path(__file__).parent),
            loaded_model_name=None,
            data_dir=os.path.join(str(Path(__file__).parent), 'data'),
            model_dir=None,
            results_dir=None,
            samples_dir=None,
            projection_dir=os.path.join(str(Path(__file__).parent),'projections'),
            remove_old_model=True,
            write_indv_pred=False,
            graph_read_format=0
        ),
        sampling_dict=dict(
            sampling_types=['kmeans', 'kmeans', 'kmeans'],
            split=[0.2, 0.25, 0.75],
            sampling_seed=112358,
            params_groups=[{
                'clusters': 5,
            }, {
                'clusters': 5,
            }, {
                'clusters': 5,
            }]
        ),
        loader_dict=dict(
            shuffle_loader=False,
            batch_size=100,
            num_workers=0,
            shuffle_steps=10
        ),
        model_dict=dict(
            n_models=1,
            num_epochs=5,
            train_delta=0.001,
            train_tolerance=1.0,
            max_deltas=4,
            loss_params={
                'function':'MaxNpercent',
                'sub_function':torch.nn.L1Loss(),
                'percent':0.1
            },
            accumulate_loss='exact',
            model=None,
            interpretable=False,
            pre_training=True,
            restart_training=False,
            optimizer_params=dict(
                dynamic_lr=False,
                optimizer='AdamW',
                params_group={
                    'lr': 0.0001
                }
            )
        )
    )
    sodas_model = SODAS(
                        mod=ALIGNN(
                            encoder=Encoder_atomic(num_species=n_types, cutoff=cutoff, dim=projection_indim, act=nn.SiLU()),
                            processor=Processor(num_convs=n_convs, dim=projection_indim, conv_type='mesh',act=nn.SiLU()),
                            decoder=Decoder(in_dim=projection_indim, out_dim=projection_outdim, act=nn.SiLU())
                        ),
                        ls_mod=umap_.UMAP(n_neighbors=50, min_dist=0.5, n_components=2),
                        pooling='softmax'
                    )

    cat = Catalyst()
    cat.set_params(parameters)

    gen_graphs = 1
    proj_data = 1
    get_gammas = 1

    if gen_graphs:
        generate_data(cat,visualize_final=True)
    if proj_data:
        cat.set_model(sodas_model)
        raw_data, projections = project_data(cat)
        if os.path.isdir(cat.parameters['io_dict']['projection_dir']):
            shutil.rmtree(cat.parameters['io_dict']['projection_dir'])
        os.makedirs(cat.parameters['io_dict']['projection_dir'])
        save_dictionary(os.path.join(cat.parameters['io_dict']['projection_dir'],'projections.data'),projections)
    if get_gammas:
        projections = load_dictionary(os.path.join(cat.parameters['io_dict']['projection_dir'],'projections.data'))
        determine_global_gammas(cat,projections)






