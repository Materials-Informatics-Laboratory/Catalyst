from pathlib import Path, PurePath
import numpy as np
import shutil
import glob
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from umap import umap_
import random
from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp
import torch as torch
from torch import nn
from catalyst.src.ml.nn.models.alignn import Encoder, Processor, Decoder,PositiveScalarsDecoder, ALIGNN
from catalyst.src.ml.testing import test_intepretable, test_non_intepretable
from catalyst.src.characterization.graph_order_parameter.gop import GOP
from catalyst.src.ml.training import run_training, run_pre_training
from catalyst.src.characterization.sodas.model.sodas import SODAS
from catalyst.src.ml.utils.distributed import cuda_destroy
from catalyst.src.graph.graph import Generic_Graph_Data
import catalyst.src.utilities.sampling as sampling
from catalyst.src.ml.ml import ML

'''
ML parameter INITIALIZATION
'''
ml_parameters = dict(
                    device_dict = dict(
                        world_size=1,
                        device='cuda',
                        run_ddp=False,
                        pin_memory=False,
                    ),
                    io_dict = dict(
                        main_path=str(Path(__file__).parent),
                        loaded_model_name=None,
                        data_dir=None,
                        model_dir=None,
                        results_dir=None,
                        samples_dir=None,
                        projection_dir=None,
                        remove_old_model=True,
                        write_indv_pred=False,
                    ),
                    sampling_dict = dict(
                            sampling_types=['kmeans','kmeans','kmeans'],
                            split=[0.1,0.2,0.75],
                            sampling_seed=112358,
                            params_groups = [{
                                'clusters':5,
                            },{
                                'clusters':5,
                            },{
                                'clusters':5,
                            }]
                    ),
                    loader_dict=dict(
                        shuffle_loader=True,
                        batch_size=[100,100,100],
                        num_workers=0,
                        shuffle_steps=10
                    ),
                    characterization_dict = dict(
                        run_characterization=True,
                        model = SODAS(mod=ALIGNN(
                            encoder=Encoder(num_species=1, cutoff=4.0, dim=100, act_func=nn.SiLU()),
                            processor=Processor(num_convs=5, dim=100),
                            decoder=Decoder(in_dim=100, out_dim=10, act_func=nn.SiLU())
                        ),
                        ls_mod=umap_.UMAP(n_neighbors=15, min_dist=0.5, n_components=2)
                        ),
                    ),
                    model_dict = dict(
                        n_models=1,
                        num_epochs=[10,10],
                        train_tolerance=1.0,
                        max_deltas=4,
                        loss_func=torch.nn.MSELoss(),
                        accumulate_loss=['sum', 'exact', 'exact'],
                        model=ALIGNN(
                            encoder=Encoder(num_species=1, cutoff=4.0, dim=100, act_func=nn.SiLU()),
                            processor=Processor(num_convs=5, dim=100, conv_type='mesh'),
                            decoder=PositiveScalarsDecoder(dim=100),
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

'''
DATA INITIALIZATION AND GRAPH CONSTRUCTION
'''
ml.parameters['io_dict']['data_dir'] = os.path.join(ml.parameters['io_dict']['main_path'],'data')
if os.path.isdir(ml.parameters['io_dict']['data_dir']):
    shutil.rmtree(ml.parameters['io_dict']['data_dir'])
os.mkdir(ml.parameters['io_dict']['data_dir'])
n_data = 1000
n_nodes = 1000
n_dim = 10
k = 5 # number of neighbors per graph nodes
dataset = []
for ds in range(n_data):
    if ds % 500 == 0:
        print('Generating graph ',ds)
    data = np.random.random((n_nodes,n_dim))
    tree = KDTree(data,metric='minkowski',leaf_size=2)
    dist, ind = tree.query(data,k=k+1) # k+1 due to self-interaction
    g_nodes = []
    g_edge_index = [[],[]]
    a_nodes = []
    for i, x in enumerate(ind):
        g_nodes.append([1])
        for j, xx in enumerate(ind[i]):
            if j > 0:
                g_edge_index[0].append(x[0])
                g_edge_index[1].append(xx)
                a_nodes.append(dist[i][j])
    g_edge_index = np.array(g_edge_index)
    graph = Generic_Graph_Data(
        node_G=torch.tensor(g_nodes, dtype=torch.float),
        edge_index_G=torch.tensor(g_edge_index, dtype=torch.long),
        node_A=torch.tensor(a_nodes, dtype=torch.float)
    )
    graph.generate_gid()
    graph.y = torch.tensor(random.uniform(0,1))
    torch.save(graph, os.path.join(os.path.join(ml.parameters['io_dict']['main_path'],ml.parameters['io_dict']['data_dir']), graph.gid + '.pt'))

'''
PROJECT DATA
'''
fig, ax = plt.subplots(nrows=1, ncols=4,sharex=True,sharey=True)
graph_data = [torch.load(file_name) for file_name in glob.glob(os.path.join(ml.parameters['io_dict']['data_dir'],'*'))]
if ml.parameters['characterization_dict']['run_characterization']:
    # read data and perform projections
    print('Performing graph projections...')
    ml.parameters['io_dict']['projection_dir'] = os.path.join(ml.parameters['io_dict']['main_path'],'projections')
    if os.path.isdir(ml.parameters['io_dict']['projection_dir']):
        shutil.rmtree(ml.parameters['io_dict']['projection_dir'])
    os.mkdir(ml.parameters['io_dict']['projection_dir'])
    ml.parameters['io_dict']['samples_dir'] = os.path.join(ml_parameters['io_dict']['main_path'], 'samples')
    if os.path.isdir(ml.parameters['io_dict']['samples_dir']):
        shutil.rmtree(ml.parameters['io_dict']['samples_dir'])
    os.mkdir(ml.parameters['io_dict']['samples_dir'])
    projected_data = None
    encoded_data = []
    gids = []
    follow_batch = ['node_G', 'node_A']
    for data in graph_data:
        gids.append(data.gid)
        loader = DataLoader([data], batch_size=1, shuffle=False, follow_batch=follow_batch,
                            num_workers=ml_parameters['loader_dict']['num_workers'])
        encoded_data.append(ml.parameters['characterization_dict']['model'].generate_gnn_latent_space(loader=loader,
                device=ml.parameters['device_dict']['device'])[0])
    encoded_data = np.array(encoded_data)
    ml.parameters['characterization_dict']['model'].fit_preprocess(data=encoded_data)
    ml.parameters['characterization_dict']['model'].fit_dim_red(data=encoded_data)
    projected_data = ml.parameters['characterization_dict']['model'].project_data(data=encoded_data)
    stored_projections = dict(
        projections=projected_data,
        gids=gids
    )
    np.save(os.path.join(ml.parameters['io_dict']['projection_dir'], 'projection_data.npy'), stored_projections)
    ax[0].plot(projected_data[:, 0], projected_data[:, 1], linestyle='', marker='o', color='w', markeredgecolor='k')
    ax[0].set_title('All data')

'''
SAMPLE DATA
'''
#start sampling
rng = np.random.default_rng(seed=ml.parameters['sampling_dict']['sampling_seed'])
# REMOVE TEST DATA
test_idx, nontest_idx = sampling.run_sampling(projected_data,
                                                  sampling_type=ml.parameters['sampling_dict']['sampling_types'][0],
                                                  split=ml.parameters['sampling_dict']['split'][0], rng=rng,
                                                  params_group=ml.parameters['sampling_dict']['params_groups'][0])
stored_test_data = dict(
        projections=[projected_data[index] for index in test_idx],
        gids=[graph_data[index].gid for index in test_idx]
)
projected_data = [projected_data[index] for index in nontest_idx]
graph_data = [graph_data[index] for index in nontest_idx]
np.save(os.path.join(ml.parameters['io_dict']['samples_dir'], 'test_data.npy'), stored_test_data)
ax[1].plot(np.array(stored_test_data['projections'])[:, 0], np.array(stored_test_data['projections'])[:, 1], linestyle='', marker='o', color='r', markeredgecolor='k')
ax[1].set_title('Test data')
# REMOVE PRETRAIN DATA
if ml.parameters['model_dict']['pre_training']:
    pretraining_data = None
    # perform pretraining
    ml.parameters['io_dict']['pretrain_dir'] = os.path.join(ml.parameters['io_dict']['samples_dir'],'pretrain')
    if os.path.isdir(ml.parameters['io_dict']['pretrain_dir']):
        shutil.rmtree(ml.parameters['io_dict']['pretrain_dir'])
    os.mkdir(ml.parameters['io_dict']['pretrain_dir'])
    # remove pretrain data
    pretrain_idx, nonpretrain_idx = sampling.run_sampling(projected_data, sampling_type=
    ml.parameters['sampling_dict']['sampling_types'][1], split=ml.parameters['sampling_dict']['split'][1], rng=rng,
                                                              params_group=ml.parameters['sampling_dict']['params_groups'][1])
    stored_pretrain_data = dict(
            training_projections=[projected_data[index] for index in pretrain_idx],
            validation_projections=None,
            training=[graph_data[index].gid for index in pretrain_idx],
            validation=None
    )
    projected_data = [projected_data[index] for index in nonpretrain_idx]
    graph_data = [graph_data[index] for index in nonpretrain_idx]
    np.save(os.path.join(ml.parameters['io_dict']['pretrain_dir'], 'train_valid_split.npy'), stored_pretrain_data,
                fix_imports=False)
    ax[2].plot(np.array(stored_pretrain_data['training_projections'])[:, 0], np.array(stored_pretrain_data['training_projections'])[:, 1], linestyle='',
                   marker='o', color='c', markeredgecolor='k')
    ax[2].set_title('Pretrain data')
# REMOVE TRAINING DATA
ml.parameters['io_dict']['model_dir'] = os.path.join(ml.parameters['io_dict']['samples_dir'], 'model_samples')
if os.path.isdir(ml.parameters['io_dict']['model_dir']):
    shutil.rmtree(ml.parameters['io_dict']['model_dir'])
os.mkdir(ml.parameters['io_dict']['model_dir'])
for iteration in range(ml.parameters['model_dict']['n_models']):
    ml.parameters['io_dict']['model_dir'] = None
    del ml.parameters['io_dict']['model_dir']
    ml.parameters['io_dict']['model_dir'] = os.path.join(ml.parameters['io_dict']['samples_dir'], 'model_samples', str(iteration))
    if os.path.isdir(ml.parameters['io_dict']['model_dir']):
        shutil.rmtree(ml.parameters['io_dict']['model_dir'])
    os.makedirs(ml.parameters['io_dict']['model_dir'], exist_ok=True)
    # sample data and train model
    train_idx, valid_idx = sampling.run_sampling(projected_data,
                                                     sampling_type=ml.parameters['sampling_dict']['sampling_types'][2],
                                                     split=ml.parameters['sampling_dict']['split'][2], rng=rng,
                                                     params_group=ml.parameters['sampling_dict']['params_groups'][2])
    train_data = [graph_data[index].gid for index in train_idx]
    valid_data = [graph_data[index].gid for index in valid_idx]
    print('Using the remaining ', len(valid_data), ' for validation')
    partitioned_data = dict(
            training_projections=[projected_data[index] for index in train_idx],
            validation_projections=[projected_data[index] for index in valid_idx],
            training=train_data,
            validation=valid_data
    )
    np.save(os.path.join(ml.parameters['io_dict']['model_dir'], 'train_valid_split.npy'), partitioned_data)
    ax[3].plot(np.array(partitioned_data ['training_projections'])[:, 0], np.array(partitioned_data['training_projections'])[:, 1],
                   linestyle='',marker='o', color='y', markeredgecolor='k')
    ax[3].set_title('Training data')
del graph_data
plt.show()

'''
PERFORM MODEL TRAINING
'''
# create model directories
ml.parameters['io_dict']['model_dir'] = None
ml.parameters['io_dict']['samples_dir'] = None
del ml.parameters['io_dict']['model_dir']
del ml.parameters['io_dict']['samples_dir']
ml.parameters['io_dict']['samples_dir'] = [os.path.join(ml_parameters['io_dict']['main_path'], 'samples','pretrain'),
                                           os.path.join(ml_parameters['io_dict']['main_path'], 'samples','model_samples')]
if ml_parameters['model_dict']['restart_training']:
    ml.parameters['io_dict']['model_dir'] = os.path.join(ml_parameters['io_dict']['main_path'], 'models_restart')
else:
    ml.parameters['io_dict']['model_dir'] = os.path.join(ml_parameters['io_dict']['main_path'], 'models')
if os.path.isdir(ml.parameters['io_dict']['model_dir']):
    shutil.rmtree(ml.parameters['io_dict']['model_dir'])
os.mkdir(ml.parameters['io_dict']['model_dir'])
if ml.parameters['model_dict']['pre_training']:
    print('Performing pretraining...')
    run_pre_training(rank=0, ml=ml)
for iteration in range(ml.parameters['model_dict']['n_models']):
    print('Performing training on model ', iteration)
    run_training(rank=0, iteration=iteration, ml=ml)
fig, ax = plt.subplots(nrows=1, ncols=2,sharex=True,sharey=False)
ax[0].set_title('Pretraining loss')
ax[1].set_title('Training loss')
fname = os.path.join(ml_parameters['io_dict']['main_path'],'models','pretraining','loss.data')
loss = []
of = open(fname,'r')
for lines in of:
    line = lines.split()
    if len(line) == 1:
        loss.append(float(line[0]))
of.close()
x=np.linspace(1,len(loss),len(loss))
ax[0].plot(x,loss,color='b',marker='o')
fname = os.path.join(ml_parameters['io_dict']['main_path'],'models','training','0','loss.data')
loss = [[],[]]
of = open(fname,'r')
for lines in of:
    line = lines.split()
    if len(line) == 2:
        loss[0].append(float(line[0]))
        loss[1].append(float(line[1]))
of.close()
x=np.linspace(1,len(loss[0]),len(loss[0]))
ax[1].plot(x,loss[0],color='b',marker='o',label='Training loss')
ax[1].plot(x,loss[1],color='r',marker='o',label='Validation loss')
ax[1].legend(loc="upper right")
plt.show()

'''
TEST MODEL
'''
ml.parameters['io_dict']['write_indv_pred'] = True
ml.parameters['io_dict']['results_dir'] = os.path.join(ml_parameters['io_dict']['main_path'],'testing')
if os.path.isdir(ml.parameters['io_dict']['results_dir']):
    shutil.rmtree(ml.parameters['io_dict']['results_dir'])
os.mkdir(ml.parameters['io_dict']['results_dir'])
model_data = torch.load(glob.glob(os.path.join(ml.parameters['io_dict']['model_dir'],'model*'))[0])
print('Testing model ' + PurePath(glob.glob(os.path.join(ml.parameters['io_dict']['model_dir'],'model*'))[0]).parts[-1])
ml.set_model()
ml.model.load_state_dict(model_data['model'])
ml.model.to(ml.parameters['device_dict']['device'])
graph_data = [torch.load(file_name) for file_name in glob.glob(os.path.join(ml.parameters['io_dict']['data_dir'],'*'))]
follow_batch = ['node_G', 'node_A', 'edge_A'] if hasattr(graph_data[0], 'edge_A') else ['node_G', 'node_A']
loader = DataLoader(graph_data, batch_size=1, shuffle=False, follow_batch=follow_batch)
print(f'Number of test graphs: {len(loader.dataset)}')
if ml.parameters['model_dict']['interpretable']:
    loss = test_intepretable(loader=loader,model=ml.model,parameters=ml.parameters)
else:
    loss = test_non_intepretable(loader=loader, model=ml.model, parameters=ml.parameters)
fig, ax = plt.subplots(nrows=1, ncols=1,sharex=True,sharey=False)
fname = os.path.join(ml_parameters['io_dict']['results_dir'],'all_indv_pred.data')
pred = [[],[]]
of = open(fname,'r')
for lines in of:
    line = lines.split()
    if len(line) == 2:
        pred[0].append(float(line[0]))
        pred[1].append(float(line[1]))
of.close()
ax.plot(pred[0],pred[0],linestyle='-',color='r')
ax.plot(pred[0],pred[1],linestyle='',color='dodgerblue',marker='o',markeredgecolor='k')
ax.set_xlabel('True values')
ax.set_ylabel('ML values')
plt.show()

























