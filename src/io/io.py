from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
import torch.distributed as dist

from ..graph.graph import Atomic_Graph_Data, Generic_Graph_Data, Graph_Data
from ..ml.utils.distributed import sync_training_dicts_across_gpus

from datetime import datetime
from pathlib import PurePath
import numpy as np
import secrets
import pickle
import random
import torch
import glob
import math
import sys
import os

import platform
import psutil
import GPUtil

def setup_dataloader(data,ml,epoch=-1,reshuffle=False,mode=0):
    parameters = ml.parameters
    if mode == 1 or mode == 0:
        if isinstance(data['training'][0], Atomic_Graph_Data):
            follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(data['training'][0], 'x_ang') else ['x_atm', 'x_bnd']
        elif isinstance(data['training'][0], Generic_Graph_Data):
            follow_batch = ['node_G', 'node_A', 'edge_A'] if hasattr(data['training'][0], 'edge_A') else ['node_G','node_A']
    elif mode == 2:
        if isinstance(data['validation'][0], Atomic_Graph_Data):
            follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(data['validation'][0], 'x_ang') else ['x_atm', 'x_bnd']
        elif isinstance(data['validation'][0], Generic_Graph_Data):
            follow_batch = ['node_G', 'node_A', 'edge_A'] if hasattr(data['validation'][0], 'edge_A') else ['node_G','node_A']

    if mode == 0:
        if reshuffle:
            if parameters['device_dict']['run_ddp']:
                train_sampler = DistributedSampler(data['training'],
                                                   shuffle=parameters['loader_dict']['shuffle_loader'],
                                                   seed=random.randint(-sys.maxsize - 1, sys.maxsize))
                train_sampler.set_epoch(epoch)
                loader_train = DataLoader(data['training'], batch_size=math.ceil(
                    parameters['loader_dict']['batch_size'][0] / parameters['device_dict']['world_size']),
                                          pin_memory=parameters['device_dict']['pin_memory'], follow_batch=follow_batch,
                                          sampler=train_sampler,
                                          num_workers=parameters['loader_dict']['num_workers'])
            else:
                loader_train = DataLoader(data['training'], pin_memory=parameters['device_dict']['pin_memory'],
                                          batch_size=parameters['loader_dict']['batch_size'][0],
                                          shuffle=parameters['loader_dict']['shuffle_loader'],
                                          follow_batch=follow_batch)
        else:
            if parameters['device_dict']['run_ddp']:
                loader_train = DataLoader(data['training'], batch_size=math.ceil(
                    parameters['loader_dict']['batch_size'][0] / parameters['device_dict']['world_size']),
                                          pin_memory=parameters['device_dict']['pin_memory'],
                                          shuffle=False, follow_batch=follow_batch,
                                          sampler=DistributedSampler(data['training']))
            else:
                loader_train = DataLoader(data['training'], pin_memory=parameters['device_dict']['pin_memory'],
                                          batch_size=parameters['loader_dict']['batch_size'][0], shuffle=True,
                                          follow_batch=follow_batch)
        return loader_train
    elif mode == 1:
        if reshuffle:
            if parameters['device_dict']['run_ddp']:
                train_sampler = DistributedSampler(data['training'],shuffle=parameters['loader_dict']['shuffle_loader'],
                                                                     seed=random.randint(-sys.maxsize - 1,sys.maxsize))
                valid_sampler = DistributedSampler(data['validation'],
                                                   shuffle=parameters['loader_dict']['shuffle_loader'],
                                                   seed=random.randint(-sys.maxsize - 1, sys.maxsize))
                train_sampler.set_epoch(epoch)
                valid_sampler.set_epoch(epoch)
                loader_train = DataLoader(data['training'], batch_size=int(
                    parameters['loader_dict']['batch_size'][1] / parameters['device_dict']['world_size']),
                                          pin_memory=parameters['device_dict']['pin_memory'],
                                          follow_batch=follow_batch,
                                          sampler=train_sampler,
                                          num_workers=parameters['loader_dict']['num_workers'])
                loader_valid = DataLoader(data['validation'],batch_size=int(
                    parameters['loader_dict']['batch_size'][2] / parameters['device_dict']['world_size']),
                                          pin_memory=parameters['device_dict']['pin_memory'],
                                          follow_batch=follow_batch,
                                          sampler=valid_sampler,
                                          num_workers=parameters['loader_dict']['num_workers'])
            else:
                loader_train = DataLoader(data['training'], pin_memory=parameters['device_dict']['pin_memory'],
                                          batch_size=parameters['loader_dict']['batch_size'][1],
                                          shuffle=parameters['loader_dict']['shuffle_loader'],
                                          follow_batch=follow_batch,
                                          num_workers=parameters['loader_dict']['num_workers'])
                loader_valid = DataLoader(data['validation'], pin_memory=parameters['device_dict']['pin_memory'],
                                          batch_size=parameters['loader_dict']['batch_size'][2],
                                          shuffle=parameters['loader_dict']['shuffle_loader'],
                                          follow_batch=follow_batch,
                                          num_workers=parameters['loader_dict']['num_workers'])
        else:
            if parameters['device_dict']['run_ddp']:
                loader_train = DataLoader(data['training'], batch_size=int(
                    parameters['loader_dict']['batch_size'][1] / parameters['device_dict']['world_size']),
                                          pin_memory=parameters['device_dict']['pin_memory'],
                                          follow_batch=follow_batch,
                                          sampler=DistributedSampler(data['training'],
                                                                     shuffle=False),
                                          num_workers=parameters['loader_dict']['num_workers'])
                loader_valid = DataLoader(data['validation'], follow_batch=follow_batch, batch_size=int(
                    parameters['loader_dict']['batch_size'][2] / parameters['device_dict']['world_size']),
                                          pin_memory=parameters['device_dict']['pin_memory'],
                                          shuffle=False, sampler=DistributedSampler(data['validation']),
                                          num_workers=parameters['loader_dict']['num_workers'])
            else:
                loader_train = DataLoader(data['training'], pin_memory=parameters['device_dict']['pin_memory'],
                                          batch_size=parameters['loader_dict']['batch_size'][1],
                                          shuffle=parameters['loader_dict']['shuffle_loader'],
                                          follow_batch=follow_batch,
                                          num_workers=parameters['loader_dict']['num_workers'])

                loader_valid = DataLoader(data['validation'], follow_batch=follow_batch,
                                          pin_memory=parameters['device_dict']['pin_memory'],
                                          batch_size=parameters['loader_dict']['batch_size'][2], shuffle=False,
                                          num_workers=parameters['loader_dict']['num_workers'])
        return loader_train, loader_valid
    elif mode == 2:
        if parameters['device_dict']['run_ddp']:
            loader_valid = DataLoader(data['validation'], follow_batch=follow_batch, batch_size=int(
                    parameters['loader_dict']['batch_size'][2] / parameters['device_dict']['world_size']),
                                          pin_memory=parameters['device_dict']['pin_memory'],
                                          shuffle=False, sampler=DistributedSampler(data['validation']),
                                          num_workers=parameters['loader_dict']['num_workers'])
        else:
            loader_valid = DataLoader(data['validation'], follow_batch=follow_batch,
                                          pin_memory=parameters['device_dict']['pin_memory'],
                                          batch_size=parameters['loader_dict']['batch_size'][2], shuffle=False,
                                          num_workers=parameters['loader_dict']['num_workers'])
        return loader_valid

def setup_model(ml,rank=0,data_only=False,load=False):
    if data_only:
        return torch.load(ml.parameters['io_dict']['loaded_model_name'])
    else:
        ml.set_model()
        ml.model.to(ml.parameters['device_dict']['device'])
        model = ml.model
        if load:
            model_data = torch.load(ml.parameters['io_dict']['loaded_model_name'])
            model.load_state_dict(model_data['model'])
        if ml.parameters['device_dict']['run_ddp']:
            model = DDP(model, device_ids=[rank],
                            find_unused_parameters=ml.parameters['device_dict']['find_unused_parameters'])
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

def save_model(model,ml,model_params_group,remove_old_models=True,pretrain=False):
    print('Saving model...')
    id = secrets.token_hex(32)
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if ml.parameters['device_dict']['run_ddp']:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    if remove_old_models:
        if pretrain:
            model_names = glob.glob(os.path.join(ml.parameters['io_dict']['model_dir'], 'pre*'))
        else:
            model_names = glob.glob(os.path.join(ml.parameters['io_dict']['model_dir'], 'model_*'))
        if len(model_names) > 0:
            for model_name in model_names:
                os.remove(model_name)
    if pretrain:
        model_data = dict(
            model_type='pretrain',
            model=model_state,
            samples=model_params_group['samples'],
            loss_info=dict(
                training=model_params_group['L_train'],
                validation=None
            ),
            id=id,
            time=str(now),
            parameters=ml.parameters,
            system_info=ml.parameters['device_dict']['system_info']
        )
        torch.save(model_data, os.path.join(ml.parameters['io_dict']['model_dir'], 'pre_' + str(id) + '_' + str(now)))
    else:
        model_data = dict(
            model_type='train',
            model=model_state,
            samples=model_params_group['samples'],
            loss_info=dict(
                training=model_params_group['L_train'],
                validation=model_params_group['L_valid']
            ),
            id=id,
            time=str(now),
            parameters=ml.parameters,
            system_info=ml.parameters['device_dict']['system_info']
        )
        torch.save(model_data,
                   os.path.join(ml.parameters['io_dict']['model_dir'], 'model_' + str(id) + '_' + str(now)))

def save_dictionary(fname,data):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dictionary(fname):
    with open(fname, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary

def write_labelled_extxyz(filename,labels,atoms,cutoff):
    of = open(filename, 'w')
    of.write(str(len(labels)) + '\n')
    of.write('Lattice="')
    for cd in atoms.get_cell():
        for c in cd:
            of.write(str(c) + ' ')
    of.write(
        ' Properties=species:S:1:pos:R:3:y:R:1 cutoff ' + str(cutoff) + ' pbc="T T T"\n')
    for j, atom in enumerate(atoms):
        of.write(atom.symbol + ' ')
        for pos in atom.position:
            of.write(str(pos) + ' ')
        of.write(str(labels[j].item()) + '\n')
    of.close()

def get_system_info():
    system_info = platform.uname()
    memory_info = psutil.virtual_memory()
    gpus = GPUtil.getGPUs()

    if not gpus:
        info = dict(
            system=system_info.system,
            node=system_info.node,
            release=system_info.release,
            version=system_info.version,
            machine=system_info.machine,
            processor=system_info.processor,
            cpu_count=psutil.cpu_count(logical=False),
            logical_count=psutil.cpu_count(logical=True),
            total_memory=memory_info.total,
            ngpus=len(gpus),
        )
    else:
        info = dict(
            system=system_info.system,
            node=system_info.node,
            release=system_info.release,
            version=system_info.version,
            machine=system_info.machine,
            processor=system_info.processor,
            cpu_count=psutil.cpu_count(logical=False),
            logical_count=psutil.cpu_count(logical=True),
            total_memory=memory_info.total,
            ngpus=len(gpus),
            gpu_type=gpus[0].name,
            gpu_driver=gpus[0].driver,
            gpu_memory=gpus[0].memoryTotal
        )
    return info

def read_training_data(params,samples_file,pretrain=False,format=0, rank=0):
    if rank == 0:
        training_graphs = None
        training_samples = None
        validation_graphs = None
        validation_samples = None

        graph_files = glob.glob(os.path.join(params['io_dict']['data_dir'],'*'))
        samples = load_dictionary(samples_file)
        training_samples = samples['training']
        if pretrain == False:
            validation_samples = samples['validation']

        if format == 0:
            gids = [PurePath(graph).parts[-1].split('.')[0] for graph in graph_files]
            if len(gids) == 0:
                print('Error: no graph files found...')
                exit(0)
        else:
            gids = [torch.load(gname)['gid'] for gname in graph_files]

        cross_list = set(training_samples).intersection(gids)
        idx = [gids.index(x) for x in cross_list]
        selected_graphs = [graph_files[i] for i in idx]
        if format == 0:
            training_graphs = [None]*len(selected_graphs)
            for i in range(len(selected_graphs)):
                training_graphs[i] = torch.load(selected_graphs[i])
        else:
            training_graphs = [None] * len(selected_graphs)
            for i in range(len(selected_graphs)):
                training_graphs[i] = selected_graphs[i]
        if pretrain == False:
            cross_list = set(validation_samples).intersection(gids)
            idx = [gids.index(x) for x in cross_list]
            selected_graphs = [graph_files[i] for i in idx]
            if format == 0:
                validation_graphs = [None] * len(selected_graphs)
                for i in range(len(selected_graphs)):
                    validation_graphs[i] = torch.load(selected_graphs[i])
            else:
                validation_graphs = [None] * len(selected_graphs)
                for i in range(len(selected_graphs)):
                    validation_graphs[i] = selected_graphs[i]

        graph_dict = dict(training=training_graphs, validation=validation_graphs)
        samples_dict = dict(training_samples=training_samples,validation_samples=validation_samples)
    else:
        graph_dict = None
        samples_dict = None
    if params['device_dict']['run_ddp']:
        dist.barrier()
        dict_list = sync_training_dicts_across_gpus(graph_dict,samples_dict)
        return dict_list[0], dict_list[1]
    else:
        return graph_dict, samples_dict

def port_graphdata_to_atomicgraphdata(path):
    graph_files = glob.glob(os.path.join(path, '*'))
    for graph in graph_files:
        data = torch.load(graph)
        if isinstance(data,Graph_Data):
            if not hasattr(data, 'atoms'):
                atoms = None
            else:
                atoms = data.atoms
            if not hasattr(data, 'y'):
                y = None
            else:
                y = data.y
            if not hasattr(data, 'x_ang'):
                x_ang = None
            else:
                x_ang = data.x_ang
            if not hasattr(data, 'mask_dih_ang'):
                mask_dih_ang = None
            else:
                mask_dih_ang = data.mask_dih_ang
            if not hasattr(data, 'ang_amounts'):
                ang_amounts = None
            else:
                ang_amounts = data.ang_amounts
            new_data = Atomic_Graph_Data(
                atoms=atoms,
                edge_index_G=data.edge_index_G,
                edge_index_A=data.edge_index_A,
                x_atm=data.x_atm,
                x_bnd=data.x_bnd,
                x_ang=x_ang,
                mask_dih_ang=mask_dih_ang,
                atm_amounts=data.atm_amounts,
                bnd_amounts=data.bnd_amounts,
                ang_amounts=ang_amounts,
            )
            new_data.y = y
            if not hasattr(data, 'gid'):
                new_data.generate_gid()
            else:
                new_data.gid = data.gid
            os.remove(graph)
            torch.save(new_data, os.path.join(path, new_data.gid + '.pt'))

