from catalyst.src.ml.nn.gnn.models.alignn import Encoder_generic,Encoder_atomic, Processor, Decoder,PositiveScalarsDecoder, ALIGNN
from catalyst.src.ml.inference import predict_external, test_non_intepretable_external, predict_interpretable
from catalyst.src.ml.training import run_training, run_pre_training
from catalyst.src.characterization.sodas.model.sodas import SODAS
from catalyst.src.graph.generic_build import generic_graph_gen
from catalyst.src.ml.utils.distributed import cuda_destroy
import catalyst.src.utilities.sampling as sampling
from catalyst.src.io.io import load_dictionary, save_dictionary
from catalyst.src.observer.params import Catalyst

from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp
import torch as torch
from torch import nn

from pathlib import Path, PurePath
import matplotlib.pyplot as plt
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
    y = []
    for i in range(regression_outdim):
        y.append(np.linspace(0,1,n_data))
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
        graph.y = []
        for i in range(regression_outdim):
            graph.y.append(torch.tensor(y[i][ds],dtype=torch.float))
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
    loader = DataLoader(graph_data, batch_size=100, shuffle=False, follow_batch=follow_batch,
                                num_workers=cat.parameters['loader_dict']['num_workers'])
    encoded_data = cat.parameters['characterization_dict']['model'].generate_gnn_latent_space(parameters=cat.parameters,loader=loader)

    encoded_data = np.array(encoded_data)
    cat.parameters['characterization_dict']['model'].fit_preprocess(data=encoded_data)
    cat.parameters['characterization_dict']['model'].fit_dim_red(data=encoded_data)
    projected_data = cat.parameters['characterization_dict']['model'].project_data(data=encoded_data)
    stored_projections = dict(
            projections=projected_data,
            gids=gids
    )
    save_dictionary(os.path.join(cat.parameters['io_dict']['projection_dir'], 'projection_data.npy'), stored_projections)
    return graph_data, projected_data

def sample_data(cat,graph_data,projected_data):
    '''
    SAMPLE DATA
    '''
    fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
    ax[0].plot(projected_data[:, 0], projected_data[:, 1], linestyle='', marker='o', color='w', markeredgecolor='k')
    ax[0].set_title('All data')
    #start sampling
    rng = np.random.default_rng(seed=cat.parameters['sampling_dict']['sampling_seed'])
    # REMOVE TEST DATA
    test_idx, nontest_idx = sampling.run_sampling(projected_data,
                                                      sampling_type=cat.parameters['sampling_dict']['sampling_types'][0],
                                                      split=cat.parameters['sampling_dict']['split'][0], rng=rng,
                                                      params_group=cat.parameters['sampling_dict']['params_groups'][0])
    stored_test_data = dict(
            projections=[projected_data[index] for index in test_idx],
            gids=[graph_data[index].gid for index in test_idx]
    )
    projected_data = [projected_data[index] for index in nontest_idx]
    graph_data = [graph_data[index] for index in nontest_idx]
    save_dictionary(os.path.join(cat.parameters['io_dict']['samples_dir'], 'test_data.npy'), stored_test_data)
    ax[1].plot(np.array(stored_test_data['projections'])[:, 0], np.array(stored_test_data['projections'])[:, 1], linestyle='', marker='o', color='r', markeredgecolor='k')
    ax[1].set_title('Test data')
    # REMOVE PRETRAIN DATA
    if cat.parameters['model_dict']['pre_training']:
        pretraining_data = None
        # perform pretraining
        cat.parameters['io_dict']['pretrain_dir'] = os.path.join(cat.parameters['io_dict']['samples_dir'],'pretrain')
        if os.path.isdir(cat.parameters['io_dict']['pretrain_dir']):
            shutil.rmtree(cat.parameters['io_dict']['pretrain_dir'])
        os.mkdir(cat.parameters['io_dict']['pretrain_dir'])
        # remove pretrain data
        pretrain_idx, nonpretrain_idx = sampling.run_sampling(projected_data, sampling_type=
        cat.parameters['sampling_dict']['sampling_types'][1], split=cat.parameters['sampling_dict']['split'][1], rng=rng,
                                                                  params_group=cat.parameters['sampling_dict']['params_groups'][1])
        stored_pretrain_data = dict(
                training_projections=[projected_data[index] for index in pretrain_idx],
                validation_projections=None,
                training=[graph_data[index].gid for index in pretrain_idx],
                validation=None
        )
        projected_data = [projected_data[index] for index in nonpretrain_idx]
        graph_data = [graph_data[index] for index in nonpretrain_idx]
        save_dictionary(os.path.join(cat.parameters['io_dict']['pretrain_dir'], 'train_valid_split.npy'), stored_pretrain_data)
        ax[2].plot(np.array(stored_pretrain_data['training_projections'])[:, 0], np.array(stored_pretrain_data['training_projections'])[:, 1], linestyle='',
                       marker='o', color='c', markeredgecolor='k')
        ax[2].set_title('Pretrain data')
    # REMOVE TRAINING DATA
    cat.parameters['io_dict']['model_dir'] = os.path.join(cat.parameters['io_dict']['samples_dir'], 'model_samples')
    if os.path.isdir(cat.parameters['io_dict']['model_dir']):
        shutil.rmtree(cat.parameters['io_dict']['model_dir'])
    os.mkdir(cat.parameters['io_dict']['model_dir'])
    for iteration in range(cat.parameters['model_dict']['n_models']):
        cat.parameters['io_dict']['model_dir'] = None
        del cat.parameters['io_dict']['model_dir']
        cat.parameters['io_dict']['model_dir'] = os.path.join(cat.parameters['io_dict']['samples_dir'], 'model_samples', str(iteration))
        if os.path.isdir(cat.parameters['io_dict']['model_dir']):
            shutil.rmtree(cat.parameters['io_dict']['model_dir'])
        os.makedirs(cat.parameters['io_dict']['model_dir'], exist_ok=True)
        # sample data and train model
        train_idx, valid_idx = sampling.run_sampling(projected_data,
                                                         sampling_type=cat.parameters['sampling_dict']['sampling_types'][2],
                                                         split=cat.parameters['sampling_dict']['split'][2], rng=rng,
                                                         params_group=cat.parameters['sampling_dict']['params_groups'][2])
        train_data = [graph_data[index].gid for index in train_idx]
        valid_data = [graph_data[index].gid for index in valid_idx]
        print('Using the remaining ', len(valid_data), ' for validation')
        partitioned_data = dict(
                training_projections=[projected_data[index] for index in train_idx],
                validation_projections=[projected_data[index] for index in valid_idx],
                training=train_data,
                validation=valid_data
        )
        save_dictionary(os.path.join(cat.parameters['io_dict']['model_dir'], 'train_valid_split.npy'), partitioned_data)
        ax[3].plot(np.array(partitioned_data ['training_projections'])[:, 0], np.array(partitioned_data['training_projections'])[:, 1],
                       linestyle='',marker='o', color='y', markeredgecolor='k')
        ax[3].set_title('Training data')
    del graph_data
    plt.show()

def train_model(cat):
    '''
    PERFORM MODEL TRAINING
    '''
    if cat.parameters['model_dict']['pre_training']:
        cat.parameters['io_dict']['samples_dir'] = None
        del cat.parameters['io_dict']['samples_dir']
        cat.parameters['io_dict']['samples_dir'] = os.path.join(cat.parameters['io_dict']['main_path'], 'samples',
                                                               'pretrain')
        print('Performing pretraining...')
        if cat.parameters['device_dict']['run_ddp']:
            processes = []
            for rank in range(cat.parameters['device_dict']['world_size']):
                p = mp.Process(target=run_pre_training, args=(rank, cat,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            cuda_destroy()
        else:
            run_pre_training(rank=0, cat=cat)
        cat.parameters['model_dict']['restart_training'] = True
        cat.parameters['io_dict']['loaded_model_name'] = None
        del cat.parameters['io_dict']['loaded_model_name']
        cat.parameters['io_dict']['loaded_model_name'] = glob.glob(os.path.join(cat.parameters['io_dict']['main_path'], 'models',
                               'pretraining', 'pre*'))[0]

    cat.parameters['io_dict']['samples_dir'] = None
    del cat.parameters['io_dict']['samples_dir']
    cat.parameters['io_dict']['samples_dir'] = os.path.join(cat.parameters['io_dict']['main_path'], 'samples','model_samples')
    for iteration in range(cat.parameters['model_dict']['n_models']):
        if cat.parameters['device_dict']['run_ddp']:
            print('Performing training on model ', iteration)
            processes = []
            for rank in range(cat.parameters['device_dict']['world_size']):
                p = mp.Process(target=run_training, args=(rank, iteration, cat,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            cuda_destroy()
        else:
            run_training(rank=0, iteration=iteration, cat=cat)
    return

def retrain_model(cat):
    cat.parameters['io_dict']['samples_dir'] = None
    del cat.parameters['io_dict']['samples_dir']
    cat.parameters['io_dict']['samples_dir'] = os.path.join(cat.parameters['io_dict']['main_path'], 'samples',
                                                           'model_samples')
    if cat.parameters['device_dict']['run_ddp']:
        print('Performing model retraining on ',cat.parameters['io_dict']['loaded_model_name'])
        processes = []
        for rank in range(cat.parameters['device_dict']['world_size']):
            p = mp.Process(target=run_training, args=(rank, 'restart', cat,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        cuda_destroy()
    else:
        run_training(rank=0, iteration='restart', cat=cat)
    return

def plot_training_results(cat,retrain=False):
    if retrain:
        fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
        ax[2].set_title('Retraining loss')
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2,sharex=True,sharey=True)
    ax[0].set_title('Pretraining loss')
    ax[1].set_title('Training loss')

    if cat.parameters['model_dict']['pre_training']:
        cat.parameters['io_dict']['model_dir'] = None
        del cat.parameters['io_dict']['model_dir']
        cat.parameters['io_dict']['model_dir'] = os.path.join(cat.parameters['io_dict']['main_path'], 'models',
                                                             'pretraining')
        run_data = load_dictionary(os.path.join(cat.parameters['io_dict']['model_dir'], 'run_information.npy'))
        loss = run_data['training_loss']
        x=np.linspace(1,len(loss),len(loss))
        ax[0].plot(x,loss,color='b',marker='o')
    loss = [[],[]]
    cat.parameters['io_dict']['model_dir'] = None
    del cat.parameters['io_dict']['model_dir']
    cat.parameters['io_dict']['model_dir'] = os.path.join(cat.parameters['io_dict']['main_path'], 'models', 'training','0')
    run_data = load_dictionary(os.path.join(cat.parameters['io_dict']['model_dir'], 'run_information.npy'))
    loss[0] = run_data['training_loss']
    loss[1] = run_data['validation_loss']
    x=np.linspace(1,len(loss[0]),len(loss[0]))
    ax[1].set_yscale('log')
    ax[1].plot(x,loss[0],color='b',marker='o',label='Training loss')
    ax[1].plot(x,loss[1],color='r',marker='o',label='Validation loss')
    ax[1].legend(loc="upper right")

    if retrain:
        loss = [[], []]
        cat.parameters['io_dict']['model_dir'] = None
        del cat.parameters['io_dict']['model_dir']
        cat.parameters['io_dict']['model_dir'] = os.path.join(cat.parameters['io_dict']['main_path'], 'models',
                                                             'training', 'restart')
        run_data = load_dictionary(os.path.join(cat.parameters['io_dict']['model_dir'], 'run_information.npy'))
        loss[0] = run_data['training_loss']
        loss[1] = run_data['validation_loss']
        x = np.linspace(1, len(loss[0]), len(loss[0]))
        ax[2].set_yscale('log')
        ax[2].plot(x, loss[0], color='b', marker='o', label='Training loss')
        ax[2].plot(x, loss[1], color='r', marker='o', label='Validation loss')
        ax[2].legend(loc="upper right")
    plt.show()


def test_model(cat):
    '''
    TEST MODEL
    '''
    cat.parameters['io_dict']['write_indv_pred'] = True
    cat.parameters['io_dict']['results_dir'] = os.path.join(cat.parameters['io_dict']['main_path'],'testing','pretraining')
    if os.path.isdir(cat.parameters['io_dict']['results_dir']):
        shutil.rmtree(cat.parameters['io_dict']['results_dir'])
    os.makedirs(cat.parameters['io_dict']['results_dir'],exist_ok=True)
    cat.parameters['io_dict']['model_dir'] = None
    del cat.parameters['io_dict']['model_dir']
    cat.parameters['io_dict']['model_dir'] = os.path.join(cat.parameters['io_dict']['main_path'],'models','pretraining')
    cat.parameters['io_dict']['loaded_model_name'] = None
    del cat.parameters['io_dict']['loaded_model_name']
    cat.parameters['io_dict']['loaded_model_name'] = glob.glob(os.path.join(cat.parameters['io_dict']['model_dir'], 'pre*'))[0]

    if cat.parameters['device_dict']['run_ddp']:
        processes = []
        for rank in range(cat.parameters['device_dict']['world_size']):
            p = mp.Process(target=test_non_intepretable_external, args=(cat,'all',rank,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        cuda_destroy()
    else:
        test_non_intepretable_external(cat,'all', rank=0)

    cat.parameters['io_dict']['results_dir'] = None
    del cat.parameters['io_dict']['results_dir']
    cat.parameters['io_dict']['results_dir'] = os.path.join(cat.parameters['io_dict']['main_path'], 'testing', 'training')
    if os.path.isdir(cat.parameters['io_dict']['results_dir']):
        shutil.rmtree(cat.parameters['io_dict']['results_dir'])
    os.makedirs(cat.parameters['io_dict']['results_dir'], exist_ok=True)
    cat.parameters['io_dict']['model_dir'] = None
    del cat.parameters['io_dict']['model_dir']
    cat.parameters['io_dict']['model_dir'] = os.path.join(cat.parameters['io_dict']['main_path'], 'models', 'training',
                                                         '0')
    cat.parameters['io_dict']['loaded_model_name'] = None
    del cat.parameters['io_dict']['loaded_model_name']
    cat.parameters['io_dict']['loaded_model_name'] = glob.glob(os.path.join(cat.parameters['io_dict']['model_dir'], 'model*'))[0]
    if cat.parameters['device_dict']['run_ddp']:
        processes = []
        for rank in range(cat.parameters['device_dict']['world_size']):
            p = mp.Process(target=test_non_intepretable_external, args=(cat,'all',rank,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        cuda_destroy()
    else:
        test_non_intepretable_external(cat,'all', rank=0)

    return

def plot_test_data(cat):
    cat.parameters['io_dict']['results_dir'] = None
    del cat.parameters['io_dict']['results_dir']
    cat.parameters['io_dict']['results_dir'] = os.path.join(cat.parameters['io_dict']['main_path'],'testing','pretraining')
    fname = os.path.join(cat.parameters['io_dict']['results_dir'],'all_indv_pred.data')
    pred = [[],[]]
    run_data = [load_dictionary(fname)]
    for i in range(len(pred)):
        for ny in range(regression_outdim):
            pred[i].append([])
    for data in run_data:
        if data['vec']:
            for data_y in data['y']:
                if data['loss_fn'] == 'sum':
                    for i, ty in enumerate(data_y):
                        pred[0][i].append(ty)
                else:
                    for i, ty in enumerate(data_y):
                        for item in ty:
                            pred[0][i].append(item)
            for data_y in data['pred']:
                if data['loss_fn'] == 'sum':
                    for i, ty in enumerate(data_y):
                        pred[1][i].append(ty)
                else:
                    for i, ty in enumerate(data_y):
                        for item in ty:
                            pred[1][i].append(item)
        else:
            for data_y in data['y']:
                if data['loss_fn'] == 'sum':
                    pred[0][0].append(data_y)
                else:
                    for i, ty in enumerate(data_y):
                        pred[0][0].append(ty)
            for data_y in data['pred']:
                if data['loss_fn'] == 'sum':
                    pred[1][0].append(data_y)
                else:
                    for i, ty in enumerate(data_y):
                        pred[1][0].append(ty)

    if len(pred[0]) > 1:
        fig, ax = plt.subplots(nrows=2, ncols=len(pred[0]), sharex=True, sharey=False)
        for i in range(len(pred[0])):
            ax[0][i].plot(pred[0][i],pred[1][i],linestyle='',color='dodgerblue',marker='o',markeredgecolor='k')
            ax[0][i].plot(pred[0][i],pred[0][i],linestyle='-',color='r')
            ax[0][i].set_xlabel('True values')
            ax[0][i].set_ylabel('ML values')
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False)
        ax[0].plot(pred[0][0], pred[1][0], linestyle='', color='dodgerblue', marker='o', markeredgecolor='k')
        ax[0].plot(pred[0][0], pred[0][0], linestyle='-', color='r')
        ax[0].set_xlabel('True values')
        ax[0].set_ylabel('ML values')
    cat.parameters['io_dict']['results_dir'] = None
    del cat.parameters['io_dict']['results_dir']
    cat.parameters['io_dict']['results_dir'] = os.path.join(cat.parameters['io_dict']['main_path'], 'testing',                                                     'training')
    fname = os.path.join(cat.parameters['io_dict']['results_dir'],'all_indv_pred.data')
    pred = [[],[]]
    run_data = [load_dictionary(fname)]
    for i in range(len(pred)):
        for ny in range(regression_outdim):
            pred[i].append([])
    for data in run_data:
        if data['vec']:
            for data_y in data['y']:
                if data['loss_fn'] == 'sum':
                    for i, ty in enumerate(data_y):
                        pred[0][i].append(ty)
                else:
                    for i, ty in enumerate(data_y):
                        for item in ty:
                            pred[0][i].append(item)
            for data_y in data['pred']:
                if data['loss_fn'] == 'sum':
                    for i, ty in enumerate(data_y):
                        pred[1][i].append(ty)
                else:
                    for i, ty in enumerate(data_y):
                        for item in ty:
                            pred[1][i].append(item)
        else:
            for data_y in data['y']:
                if data['loss_fn'] == 'sum':
                    pred[0][0].append(data_y)
                else:
                    for i, ty in enumerate(data_y):
                        pred[0][0].append(ty)
            for data_y in data['pred']:
                if data['loss_fn'] == 'sum':
                    pred[1][0].append(data_y)
                else:
                    for i, ty in enumerate(data_y):
                        pred[1][0].append(ty)
    if len(pred[0]) > 1:
        for i in range(len(pred[0])):
            ax[1][i].plot(pred[0][i], pred[1][i], linestyle='', color='dodgerblue', marker='o', markeredgecolor='k')
            ax[1][i].plot(pred[0][i], pred[0][i], linestyle='-', color='r')
            ax[1][i].set_xlabel('True values')
            ax[1][i].set_ylabel('ML values')
    else:
        ax[1].plot(pred[0][0], pred[1][0], linestyle='', color='dodgerblue', marker='o', markeredgecolor='k')
        ax[1].plot(pred[0][0], pred[0][0], linestyle='-', color='r')
        ax[1].set_xlabel('True values')
        ax[1].set_ylabel('ML values')
    plt.show()

def predict(cat,interpret):
    cat.parameters['io_dict']['write_indv_pred'] = False
    cat.parameters['io_dict']['results_dir'] = None
    del cat.parameters['io_dict']['results_dir']
    cat.parameters['io_dict']['results_dir'] = os.path.join(cat.parameters['io_dict']['main_path'], 'testing', 'predict')
    if os.path.isdir(cat.parameters['io_dict']['results_dir']):
        shutil.rmtree(cat.parameters['io_dict']['results_dir'])
    os.makedirs(cat.parameters['io_dict']['results_dir'], exist_ok=True)
    cat.parameters['io_dict']['model_dir'] = None
    del cat.parameters['io_dict']['model_dir']
    cat.parameters['io_dict']['model_dir'] = os.path.join(cat.parameters['io_dict']['main_path'], 'models', 'training',
                                                         '0')
    cat.parameters['io_dict']['loaded_model_name'] = None
    del cat.parameters['io_dict']['loaded_model_name']
    cat.parameters['io_dict']['loaded_model_name'] = \
    glob.glob(os.path.join(cat.parameters['io_dict']['model_dir'], 'model*'))[0]
    if cat.parameters['device_dict']['run_ddp']:
        processes = []
        for rank in range(cat.parameters['device_dict']['world_size']):
            p = mp.Process(target=predict_external, args=(cat, 'all', rank,interpret))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        cuda_destroy()
    else:
        predict_external(cat, 'all', rank=0,interpretable=interpret)

    return

if __name__ == '__main__':
    n_types = 4  # number of ficticious types to label each node in G
    projection_indim = 100
    projection_outdim = 100
    regression_indim = 100
    regression_outdim = 1
    cutoff = 10.0
    n_convs = 3
    n_data = 1000 # total number of samples
    n_nodes = np.linspace(10, 100, n_data)  # number of data points per sample
    n_dim = 10  # number of dimensions in intial raw data
    parameters = dict(
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
            batch_size=[10,100,100],
            num_workers=0,
            shuffle_steps=10
        ),
        characterization_dict=dict(
            model=SODAS(mod=ALIGNN(
                encoder=Encoder_atomic(num_species=n_types, cutoff=cutoff, dim=projection_indim, act=nn.SiLU()),
                processor=Processor(num_convs=n_convs, dim=projection_indim, conv_type='mesh',act=nn.SiLU()),
                decoder=Decoder(in_dim=projection_indim, out_dim=projection_outdim, act=nn.SiLU())
            ),
                ls_mod=umap_.UMAP(n_neighbors=10, min_dist=0.1, n_components=2)
            ),
        ),
        model_dict=dict(
            n_models=1,
            num_epochs=[2,250],
            train_delta=[0.01, 0.001],
            train_tolerance=[1.0, 0.0001],
            max_deltas=4,
            loss_params={
                'function':'MaxNpercent',
                'sub_function':torch.nn.L1Loss(),
                'percent':0.1
            },
            accumulate_loss=['exact', 'exact', 'exact'],
            model=ALIGNN(
                encoder=Encoder_atomic(num_species=n_types, cutoff=cutoff, dim=regression_indim, act=nn.SiLU()),
                processor=Processor(num_convs=n_convs, dim=regression_indim, conv_type='mesh', act=nn.SiLU()),
                decoder=PositiveScalarsDecoder(dim=regression_indim, act=nn.SiLU()),
                #decoder=Decoder(in_dim=regression_indim, out_dim=regression_outdim, act=nn.SiLU(),combine=False)
            ),
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
    cat = Catalyst()
    cat.set_params(parameters)

    gen_graphs =0
    project_graphs =0
    gen_samples = 0
    perform_train = 1
    perform_retrain = 0
    perform_test = 1
    plot_test = 1
    plot_training =0
    perform_ranking = 0
    perform_predictions = 0

    if gen_graphs:
        generate_data(cat,visualize_final=True)
    if project_graphs:
        cat.parameters['characterization_dict']['model'].send_model()
        raw_data, projections = project_data(cat)
        cat.parameters['characterization_dict']['model'].clear_model()
    if gen_samples:
        sample_data(cat,graph_data=raw_data,projected_data=projections)
    if perform_train:
        train_model(cat)
        if perform_retrain:
            cat.parameters['model_dict']['restart_training'] = True
            cat.parameters['io_dict']['loaded_model_name'] = None
            del cat.parameters['io_dict']['loaded_model_name']
            cat.parameters['io_dict']['loaded_model_name'] = \
            glob.glob(os.path.join(cat.parameters['io_dict']['main_path'], 'models',
                                   'training', '0', 'model*'))[0]
            retrain_model(cat)
        if plot_training:
            plot_training_results(cat,retrain=perform_retrain)
    elif perform_retrain:
        cat.parameters['model_dict']['restart_training'] = True
        cat.parameters['io_dict']['loaded_model_name'] = None
        del cat.parameters['io_dict']['loaded_model_name']
        cat.parameters['io_dict']['loaded_model_name'] = \
        glob.glob(os.path.join(cat.parameters['io_dict']['main_path'], 'models',
                                   'training', '0', 'model*'))[0]
        retrain_model(cat)
    if perform_test:
        test_model(cat)
    if plot_test:
        plot_test_data(cat)
    if perform_predictions:
        predict(cat,perform_ranking)




















