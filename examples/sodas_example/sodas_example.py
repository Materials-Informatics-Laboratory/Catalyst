from catalyst.src.ml.nn.models.alignn import Encoder_atomic,Processor, Decoder,PositiveScalarsDecoder, ALIGNN
from catalyst.src.ml.training import run_training, run_pre_training
from catalyst.src.characterization.sodas.model.sodas import SODAS
from catalyst.src.graph.generic_build import generic_pairwise
import catalyst.src.utilities.sampling as sampling
from catalyst.src.io.io import load_dictionary, save_dictionary
from catalyst.src.ml.ml import ML
from catalyst.src.characterization.sodas.utils.utils import generate_latent_space_path, assign_gammas

from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp
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
global regression_indim
global regression_outdim
global n_convs
global n_data
global n_nodes
global n_dim


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

def generate_data(ml,visualize_final=False):
    '''
    DATA INITIALIZATION AND GRAPH CONSTRUCTION
    '''
    if os.path.isdir(ml.parameters['io_dict']['data_dir']):
        shutil.rmtree(ml.parameters['io_dict']['data_dir'])
    os.mkdir(ml.parameters['io_dict']['data_dir'])

    k = np.linspace(3,9,n_data) # number of neighbors per graph node
    dataset = []
    y = []
    scale = 1
    for i in range(regression_outdim):
        y.append(np.linspace(0,1,int(n_data/scale)))
    '''
    for i in range(regression_outdim):
        while len(y[i]) < n_data:
            y[i] = np.append(y[i],1.0)
    '''
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
        graph = generic_pairwise(data,data_params=neighbor_data,gen_line_graph=False)
        graph.y = []
        for i in range(regression_outdim):
            graph.y.append(torch.tensor(y[i][ds],dtype=torch.float))
        torch.save(graph, os.path.join(os.path.join(ml.parameters['io_dict']['main_path'],ml.parameters['io_dict']['data_dir']), str(ds) + '.pt'))
    if visualize_final:
        visualize_graph(graph, atomic=False)

def project_data(ml):
    '''
    PROJECT DATA
    '''
    graph_data = []
    for ds in range(n_data):
        graph_data.append(torch.load(os.path.join(ml.parameters['io_dict']['data_dir'],str(ds) + '.pt')))
    # read data and perform projections
    print('Performing graph projections...')
    ml.parameters['io_dict']['projection_dir'] = os.path.join(ml.parameters['io_dict']['main_path'],'projections')
    if os.path.isdir(ml.parameters['io_dict']['projection_dir']):
        shutil.rmtree(ml.parameters['io_dict']['projection_dir'])
    os.mkdir(ml.parameters['io_dict']['projection_dir'])
    ml.parameters['io_dict']['samples_dir'] = os.path.join(ml.parameters['io_dict']['main_path'], 'samples')
    if os.path.isdir(ml.parameters['io_dict']['samples_dir']):
        shutil.rmtree(ml.parameters['io_dict']['samples_dir'])
    os.mkdir(ml.parameters['io_dict']['samples_dir'])
    projected_data = None
    encoded_data = []
    gids = []
    y = []
    follow_batch = ['node_G', 'node_A', 'edge_A'] if hasattr(graph_data[0], 'edge_A') else ['node_G', 'node_A']
    for data in graph_data:
        gids.append(data.gid)
        y.append(data.y)
        loader = DataLoader([data], batch_size=1, shuffle=False, follow_batch=follow_batch,
                                num_workers=ml.parameters['loader_dict']['num_workers'])
        encoded_data.append(ml.parameters['characterization_dict']['model'].generate_gnn_latent_space(loader=loader,
                    device=ml.parameters['device_dict']['device'])[0])
    encoded_data = np.array(encoded_data)
    ml.parameters['characterization_dict']['model'].fit_preprocess(data=encoded_data)
    ml.parameters['characterization_dict']['model'].fit_dim_red(data=encoded_data)
    projected_data = ml.parameters['characterization_dict']['model'].project_data(data=encoded_data)
    stored_projections = dict(
            projections=projected_data,
            y=y,
            gids=gids,
            graphs = graph_data
    )
    save_dictionary(os.path.join(ml.parameters['io_dict']['projection_dir'], 'global_info.data'), stored_projections)

    return stored_projections

if __name__ == '__main__':
    n_types = 3  # number of ficticious types to label each node in G
    projection_indim = 10
    projection_outdim = 10
    regression_indim = 10
    regression_outdim = 1
    cutoff = 4.0
    n_convs = 3
    n_data = 2000 # total number of samples
    n_nodes = np.linspace(10, 100, n_data)  # number of data points per sample
    n_dim = 5  # number of dimensions in intial raw data
    ml_parameters = dict(
        device_dict=dict(
            world_size=2,
            device='cuda',
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
            projection_dir=None,
            remove_old_model=True,
            write_indv_pred=False,
            graph_read_format=0
        ),
        sampling_dict=dict(
            sampling_types=['kmeans', 'kmeans', 'kmeans'],
            split=[0.01, 0.01, 0.75],
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
            batch_size=[200,200,300],
            num_workers=0,
            shuffle_steps=10
        ),
        characterization_dict=dict(
            model=SODAS(mod=ALIGNN(
                encoder=Encoder_atomic(num_species=n_types, cutoff=cutoff, dim=projection_indim, act_func=nn.SiLU()),
                processor=Processor(num_convs=n_convs, dim=projection_indim, conv_type='mesh'),
                decoder=Decoder(in_dim=projection_indim, out_dim=projection_outdim, act_func=nn.SiLU())
            ),
                ls_mod=umap_.UMAP(n_neighbors=20, min_dist=0.5, n_components=2)
            ),
        ),
        model_dict=dict(
            n_models=1,
            num_epochs=[10,10],
            train_delta=[0.01, 0.001],
            train_tolerance=[1.0, 0.0001],
            max_deltas=4,
            loss_func=torch.nn.MSELoss(),
            accumulate_loss=['sum', 'exact', 'exact'],
            model=ALIGNN(
                encoder=Encoder_atomic(num_species=n_types, cutoff=cutoff, dim=regression_indim, act_func=nn.SiLU()),
                processor=Processor(num_convs=n_convs, dim=regression_indim, conv_type='mesh'),
                decoder=PositiveScalarsDecoder(dim=regression_indim, act_func=nn.SiLU()),
            ),
            interpretable=False,
            pre_training=True,
            restart_training=False,
            optimizer_params=dict(
                dynamic_lr=False,
                optimizer='AdamW',
                params_group={
                    'lr': 0.001
                }
            )
        )
    )
    ml = ML()
    ml.set_params(ml_parameters)

    gen_data = 1
    proj_data = 1

    if gen_data:
        generate_data(ml,False)
    if proj_data:
        global_data = project_data(ml)
    else:
        ml.parameters['io_dict']['projection_dir'] = os.path.join(ml.parameters['io_dict']['main_path'], 'projections')
        global_data = load_dictionary(os.path.join(ml.parameters['io_dict']['projection_dir'], 'global_info.data'))

    fig, ax = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)


    path_data = generate_latent_space_path(global_data['projections'], boundaries=[100, -100], k=5)
    assigned_gammas = assign_gammas(global_data['projections'],global_data['projections'], path_data,version='2',k=5,
                                    iterations=3, scale=0.1,cutoff=20.0
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
    for i in range(len(assigned_gammas)):
        graph = torch.load(os.path.join(ml.parameters['io_dict']['data_dir'],str(i) + '.pt'))
        delta = graph['y'][0].item() - assigned_gammas[i]
        deltas.append(delta)

    ax[0][3].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=deltas,
                     cmap='bwr',edgecolor='k')
    title = str(np.average(np.array(deltas))) + ' ' + str(np.std(np.array(deltas)))
    ax[0][3].set_title(title)

    import matplotlib.colors as mcolors

    norm = mcolors.Normalize(vmin=min(deltas), vmax=max(deltas))
    mappable = cm.ScalarMappable(norm=norm, cmap='bwr')
    fig.colorbar(mappable, ax=ax[0][3])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[0][2])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[0][0])

    path_data = generate_latent_space_path(global_data['projections'], boundaries=[100, -100], k=7,version='2',reduction=0.1)
    assigned_gammas = assign_gammas(global_data['projections'], global_data['projections'], path_data,version='2',k=2,
                                    iterations=2, scale=0.1,cutoff=20.0
                                    )

    ax[1][0].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=global_data['y'])

    ax[1][1].plot(path_data['weighted_path'][:,0],
                  path_data['weighted_path'][:,1], color='k',markeredgecolor='r', marker='o',
                  linestyle='-', markerfacecolor='w')
    ax[1][1].scatter(path_data['weighted_path'][0][0],
                     path_data['weighted_path'][0][1], c='b', s=250)
    ax[1][1].scatter(path_data['weighted_path'][-1][0],
                     path_data['weighted_path'][-1][1], c='k', s=250)

    ax[1][2].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=assigned_gammas)

    deltas = []
    for i in range(len(assigned_gammas)):
        graph = torch.load(os.path.join(ml.parameters['io_dict']['data_dir'], str(i) + '.pt'))
        delta = graph['y'][0].item() - assigned_gammas[i]
        deltas.append(delta)

    ax[1][3].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=deltas,
                     cmap='bwr', edgecolor='k')
    title = str(np.average(np.array(deltas))) + ' ' + str(np.std(np.array(deltas)))
    ax[1][3].set_title(title)

    norm = mcolors.Normalize(vmin=min(deltas), vmax=max(deltas))
    mappable = cm.ScalarMappable(norm=norm, cmap='bwr')
    fig.colorbar(mappable, ax=ax[1][3])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[1][2])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[1][0])

    path_data = generate_latent_space_path(global_data['projections'], boundaries=[100, -100], k=7, version='3',
                                           reduction=0.1)
    assigned_gammas = assign_gammas(global_data['projections'], global_data['projections'], path_data, k=2,
                                    version='2',
                                    iterations=2, scale=0.1,cutoff=20.0
                                    )

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
    for i in range(len(assigned_gammas)):
        graph = torch.load(os.path.join(ml.parameters['io_dict']['data_dir'], str(i) + '.pt'))
        delta = graph['y'][0].item() - assigned_gammas[i]
        deltas.append(delta)

    ax[2][3].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=deltas,
                     cmap='bwr', edgecolor='k')
    title = str(np.average(np.array(deltas))) + ' ' + str(np.std(np.array(deltas)))
    ax[2][3].set_title(title)

    norm = mcolors.Normalize(vmin=min(deltas), vmax=max(deltas))
    mappable = cm.ScalarMappable(norm=norm, cmap='bwr')
    fig.colorbar(mappable, ax=ax[2][3])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[2][2])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[2][0])

    assigned_gammas = assign_gammas(global_data['projections'], global_data['projections'], path_data, k=2,version='2',
                                    iterations=2,scale=0.1,cutoff=20.0)

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
    for i in range(len(assigned_gammas)):
        graph = torch.load(os.path.join(ml.parameters['io_dict']['data_dir'], str(i) + '.pt'))
        delta = graph['y'][0].item() - assigned_gammas[i]
        deltas.append(delta)

    ax[3][3].scatter(global_data['projections'][:, 0], global_data['projections'][:, 1], c=deltas,
                     cmap='bwr', edgecolor='k')
    title = str(np.average(np.array(deltas))) + ' ' + str(np.std(np.array(deltas)))
    ax[3][3].set_title(title)

    norm = mcolors.Normalize(vmin=min(deltas), vmax=max(deltas))
    mappable = cm.ScalarMappable(norm=norm, cmap='bwr')
    fig.colorbar(mappable, ax=ax[3][3])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[3][2])
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=ax[3][0])

    plt.show()


















