from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group,destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
import torch.distributed as dist
from numba import cuda
from torch import nn
import torch

from .testing import test_intepretable, test_non_intepretable
from ..utilities.distributions import get_distribution
from .utils.distributed import ddp_destroy, ddp_setup, reduce_tensor
from .utils.optimizer import set_optimizer
from .utils.memory import optimizer_to
from ..io.io import read_training_data

from datetime import datetime
import numpy as np
import random
import shutil
import glob
import time
import sys
import os
import gc

def accumulate_predictions(pred,data,loss_tag):
    if loss_tag == 'exact':
        if hasattr(data, 'x_atm_batch'): #atomic graph
            d = pred[0].flatten()
            x = torch.unique(data['x_atm_batch'])
            sorted_atms = [None] * len(x)
            dd = data['x_atm_batch']
            for i, xx in enumerate(x):
                xw = torch.where(dd == xx)
                sorted_atms[i] = d[xw]
            d = pred[1].flatten()
            x = torch.unique(data['x_bnd_batch'])
            sorted_bnds = [None] * len(x)
            dd = data['x_bnd_batch']
            for i, xx in enumerate(x):
                xw = torch.where(dd == xx)
                sorted_bnds[i] = d[xw]
            if hasattr(data, 'x_ang_batch'):
                d = pred[2].flatten()
                x = torch.unique(data['x_ang_batch'])
                sorted_angs = [None] * len(x)
                dd = data['x_ang_batch']
                for i, xx in enumerate(x):
                    xw = torch.where(dd == xx)
                    sorted_angs[i] = d[xw]
            preds = []
            for i in range(len(sorted_atms)):
                if hasattr(data, 'x_ang_batch'):
                    preds.append(sorted_atms[i].sum() + sorted_angs[i].sum() + sorted_bnds[i].sum())
                else:
                    preds.append(sorted_atms[i].sum() + sorted_bnds[i].sum())
            preds = torch.stack(preds)
        if hasattr(data, 'node_G_batch'): # generic graph
            d = pred[0].flatten()
            x = torch.unique(data['node_G_batch'])
            sorted_nodes_G = [None] * len(x)
            dd = data['node_G_batch']
            for i, xx in enumerate(x):
                xw = torch.where(dd == xx)
                sorted_nodes_G[i] = d[xw]
            d = pred[1].flatten()
            x = torch.unique(data['node_A_batch'])
            sorted_nodes_A = [None] * len(x)
            dd = data['node_A_batch']
            for i, xx in enumerate(x):
                xw = torch.where(dd == xx)
                sorted_nodes_A[i] = d[xw]
            if hasattr(data, 'edge_A_batch'):
                d = pred[2].flatten()
                x = torch.unique(data['edge_A_batch'])
                sorted_edges_A = [None] * len(x)
                dd = data['edge_A_batch']
                for i, xx in enumerate(x):
                    xw = torch.where(dd == xx)
                    sorted_edges_A[i] = d[xw]
            preds = []
            for i in range(len(sorted_nodes_G)):
                if hasattr(data, 'edge_A_batch'):
                    preds.append(sorted_nodes_G[i].sum() + sorted_nodes_A[i].sum() + sorted_edges_A[i].sum())
                else:
                    preds.append(sorted_nodes_G[i].sum() + sorted_nodes_A[i].sum())
            preds = torch.stack(preds)
        y = data.y.flatten()
    elif loss_tag == 'sum':
        preds = None
        for p in pred:
            if preds is None:
                preds = p.sum()
            else:
                preds += p.sum()
        y = data.y.flatten().sum()

    return preds, y

def train(loader,model,parameters,optimizer,pretrain=False):
    model.train()
    if pretrain:
        loss_accum = parameters['model_dict']['accumulate_loss'][0]
    else:
        loss_accum = parameters['model_dict']['accumulate_loss'][1]
    total_loss = 0.0
    if parameters['device_dict']['run_ddp'] == False:
        model.to(parameters['device_dict']['device'])
    loss_fn = parameters['model_dict']['loss_func']
    for data in loader:
        def closure():
            data.to(parameters['device_dict']['device'], non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            pred = model(data)
            preds, y = accumulate_predictions(pred,data,loss_accum)
            loss = loss_fn(preds,y)
            nonlocal total_loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            return loss
        optimizer.step(closure)
    if parameters['device_dict']['run_ddp']:
        total_loss = reduce_tensor(torch.tensor(total_loss).to(parameters['device_dict']['device'])).item()
    return total_loss / (len(loader)*parameters['device_dict']['world_size'])

def run_training(rank,iteration,ml=None):

    parameters = ml.parameters
    if parameters['device_dict']['run_ddp']:
        ddp_setup(rank, parameters['device_dict']['world_size'], parameters['device_dict']['ddp_backend'])
    #for iteration in range(parameters['model_dict']['n_models']):
    parameters['io_dict']['model_save_dir'] = os.path.join(parameters['io_dict']['model_dir'], str(iteration))
    if rank == 0:
        if os.path.isdir(parameters['io_dict']['model_save_dir']):
            shutil.rmtree(parameters['io_dict']['model_save_dir'])
        os.mkdir(parameters['io_dict']['model_save_dir'])
        print('Reading data...')

    data = read_training_data(parameters,
                              os.path.join(parameters['io_dict']['samples_dir'], 'model_samples', str(iteration), 'train_valid_split.npy'))
    if rank == 0:
        print('Training model ', iteration)
        print('Training using ',len(data['training']), ' training points and ',len(data['validation']),' validation points...')
        stats_file = open(os.path.join(parameters['io_dict']['model_save_dir'], 'loss.data'), 'w')
        stats_file.write('Training_loss     Validation loss\n')
        stats_file.close()

    ml.set_model()
    ml.model.to(ml.parameters['device_dict']['device'])
    model = ml.model
    if parameters['model_dict']['pre_training'] == False:
        if parameters['model_dict']['restart_training']:
            print('Restarting model training using model ', parameters['io_dict']['restart_model_name'])
            model.load_state_dict(torch.load(ml.parameters['io_dict']['restart_model_name'],map_location=ml.parameters['device_dict']['device']))
        if parameters['device_dict']['run_ddp']:
            model = DDP(model, device_ids=[rank],find_unused_parameters=True)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model_name = glob.glob(os.path.join(parameters['io_dict']['pretrain_dir'], 'model_*'))
        model.load_state_dict(torch.load(model_name[0],map_location=ml.parameters['device_dict']['device']))
        if parameters['device_dict']['run_ddp']:
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(data['training'][0], 'x_ang') else ['x_atm','x_bnd']
    if parameters['device_dict']['run_ddp']:
        loader_train = DataLoader(data['training'], batch_size=int(
                parameters['loader_dict']['batch_size'][1] / parameters['device_dict']['world_size']),
                                      pin_memory=parameters['device_dict']['pin_memory'],
                                      follow_batch=follow_batch,
                                      sampler=DistributedSampler(data['training'],
                                                                 shuffle=False),num_workers=parameters['loader_dict']['num_workers'])
        loader_valid = DataLoader(data['validation'], follow_batch=follow_batch,batch_size=int(
                        parameters['loader_dict']['batch_size'][2] / parameters['device_dict']['world_size']),pin_memory=parameters['device_dict']['pin_memory'],
                                      shuffle=False, sampler=DistributedSampler(data['validation']),num_workers=parameters['loader_dict']['num_workers'])
    else:
        loader_train = DataLoader(data['training'], pin_memory=parameters['device_dict']['pin_memory'],
                                      batch_size=parameters['loader_dict']['batch_size'][1],
                                      shuffle=parameters['loader_dict']['shuffle_loader'],
                                      follow_batch=follow_batch,
                                      num_workers=parameters['loader_dict']['num_workers'])

        loader_valid = DataLoader(data['validation'],follow_batch=follow_batch,pin_memory=parameters['device_dict']['pin_memory'],
                                      batch_size=parameters['loader_dict']['batch_size'][2], shuffle=False,num_workers=parameters['loader_dict']['num_workers'])

    if parameters['model_dict']['optimizer_params']['dynamic_lr']:
        dist_params = dict(
                dist_type=parameters['model_dict']['optimizer_params']['dist_type'],
                vars=parameters['model_dict']['optimizer_params']['lr_scale'],
                size=parameters['model_dict']['num_epochs'][1],
                floor=parameters['model_dict']['optimizer_params']['params_group']['lr']
            )
        lr_data = get_distribution(dist_params)
    else:
        lr_data = np.linspace(parameters['model_dict']['optimizer_params']['params_group']['lr'],
                                  parameters['model_dict']['optimizer_params']['params_group']['lr'],parameters['model_dict']['num_epochs'][1])

    epoch_times = []
    running_valid_delta = []
    L_train, L_valid = [], []
    min_loss_train = 1.0E30
    min_loss_valid = 1.0E30
    ep = 0
    while ep < parameters['model_dict']['num_epochs'][1]:
        if rank == 0:
            start_time = time.time()
            print('Epoch ',ep+1,' of ',parameters['model_dict']['num_epochs'][1],  ' lr_rate: ',lr_data[ep], 'loss_accum: ',parameters['model_dict']['accumulate_loss'][1])
            sys.stdout.flush()

        if parameters['loader_dict']['shuffle_loader'] == True: #reshuffle training data to avoid overfitting
            if ep % parameters['loader_dict']['shuffle_steps'] == 0 and ep > 0:
                if rank == 0:
                    print('Shuffling training data...')
                if parameters['device_dict']['run_ddp']:
                    loader_train.sampler.set_epoch(ep)
                    loader_valid.sampler.set_epoch(ep)
                    loader_train = DataLoader(data['training'], batch_size=int(
                            parameters['loader_dict']['batch_size'][1] / parameters['device_dict']['world_size']),
                                                  pin_memory=parameters['device_dict']['pin_memory'],
                                                  follow_batch=follow_batch,
                                                  sampler=DistributedSampler(data['training'],shuffle=parameters['loader_dict']['shuffle_loader'],
                                                                             seed=random.randint(-sys.maxsize - 1, sys.maxsize)),
                                                  num_workers=parameters['loader_dict']['num_workers'])
                else:
                    loader_train = DataLoader(data['training'], pin_memory=parameters['device_dict']['pin_memory'],
                                              batch_size=parameters['loader_dict']['batch_size'][1],
                                              shuffle=parameters['loader_dict']['shuffle_loader'],
                                              follow_batch=follow_batch,
                                              num_workers=parameters['loader_dict']['num_workers'])

        parameters['model_dict']['optimizer_params']['params_group']['lr'] = lr_data[ep]
        if parameters['device_dict']['run_ddp']:
            parameters['model_dict']['optimizer_params']['params_group'][
                    'params'] = model.module.processor.parameters()
        else:
            parameters['model_dict']['optimizer_params']['params_group']['params'] = model.processor.parameters()

        optimizer = set_optimizer(parameters)
        optimizer_to(optimizer,parameters['device_dict']['device'])

        loss_train = train(loader_train, model, parameters, optimizer);
        loss_valid = test_non_intepretable(loader_valid, model, parameters)

        if rank == 0:
            epoch_times.append(time.time() - start_time)
            print('epoch_time = ', time.time() - start_time, ' seconds Average epoch time = ', sum(epoch_times) / float(len(epoch_times)), ' seconds')
            print('Train loss = ',loss_train,' Validation loss = ',loss_valid)
            stats_file = open(os.path.join(parameters['io_dict']['model_save_dir'], 'loss.data'), 'a')
            stats_file.write(str(loss_train) + '     ' + str(loss_valid) + '\n')
            stats_file.close()
        if ep > 0:
            delta_val = loss_valid - L_valid[-1]
            running_valid_delta.append(abs(delta_val))
            if len(running_valid_delta) > parameters['model_dict']['max_deltas']:
                running_valid_delta.pop(0)
                if rank == 0:
                    print('Running validation delta = ',sum(running_valid_delta)/len(running_valid_delta))

        L_train.append(loss_train)
        L_valid.append(loss_valid)

        if loss_train < min_loss_train:
            min_loss_train = loss_train
            if loss_valid < min_loss_valid:
                min_loss_valid = loss_valid
                if rank == 0:
                     if parameters['io_dict']['remove_old_model']:
                        model_name = glob.glob(os.path.join(parameters['io_dict']['model_save_dir'], 'model_*'))
                        if len(model_name) > 0 and rank == 0:
                            os.remove(model_name[0])
                     print('Saving model...')
                     now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                     if parameters['device_dict']['run_ddp']:
                        torch.save(model.module.state_dict(), os.path.join(parameters['io_dict']['model_save_dir'], 'model_' + str(now)+'_ep-'+str(ep)))
                     else:
                        torch.save(model.state_dict(), os.path.join(parameters['io_dict']['model_save_dir'], 'model_' + str(now)+'_ep-'+str(ep)))
        if len(running_valid_delta) == parameters['model_dict']['max_deltas']:
            if sum(running_valid_delta)/len(running_valid_delta)< parameters['model_dict']['train_tolerance']:
                if rank == 0:
                    print('Validation delta satisfies set tolerance...exiting training loop...')
                ep = parameters['model_dict']['num_epochs'][1]
        ep += 1
    if parameters['device_dict']['run_ddp']:
        ddp_destroy()

def run_pre_training(rank,ml=None):
    parameters = ml.parameters
    ml.set_model()
    model = ml.model
    if rank == 0:
        if os.path.isdir(parameters['io_dict']['pretrain_dir']):
            shutil.rmtree(parameters['io_dict']['pretrain_dir'])
        os.mkdir(parameters['io_dict']['pretrain_dir'])
        print('Reading graphs...')
    data = read_training_data(parameters,os.path.join(parameters['io_dict']['samples_dir'], 'pre_training', 'train_valid_split.npy'),pretrain=True)
    if rank == 0:
        print('Training using ', len(data['training']), ' training points')
        stats_file = open(os.path.join(parameters['io_dict']['pretrain_dir'], 'loss.data'), 'w')
        stats_file.write('Training_loss\n')
        stats_file.close()

    if parameters['device_dict']['run_ddp']:
        ddp_setup(rank, parameters['device_dict']['world_size'],parameters['device_dict']['ddp_backend'])
        model.to(parameters['device_dict']['device'])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank],find_unused_parameters=True)

    follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(data['training'][0], 'x_ang') else ['x_atm'] # need to rework this
    if parameters['device_dict']['run_ddp']:
        loader_train = DataLoader(data['training'], batch_size=int(
                        parameters['loader_dict']['batch_size'][0] / parameters['device_dict']['world_size']),
                                  pin_memory=parameters['device_dict']['pin_memory'],
                                  shuffle=False, follow_batch=follow_batch,
                                  sampler=DistributedSampler(data['training']))
    else:
        loader_train = DataLoader(data['training'], pin_memory=parameters['device_dict']['pin_memory'],
                                  batch_size=parameters['loader_dict']['batch_size'][0], shuffle=True, follow_batch=follow_batch)

    if parameters['model_dict']['optimizer_params']['dynamic_lr']:
        dist_params = dict(
            dist_type=parameters['model_dict']['optimizer_params']['dist_type'],
            vars=parameters['model_dict']['optimizer_params']['lr_scale'],
            size=parameters['model_dict']['num_epochs'][1],
            floor=parameters['model_dict']['optimizer_params']['params_group']['lr']
        )
        lr_data = get_distribution(dist_params)
    else:
        lr_data = np.linspace(parameters['model_dict']['optimizer_params']['params_group']['lr'],
                              parameters['model_dict']['optimizer_params']['params_group']['lr'],
                              parameters['model_dict']['num_epochs'][1])

    epoch_times = []
    running_train_delta = []
    L_train = []
    min_loss_train = 1.0E30
    ep = 0
    while ep < parameters['model_dict']['num_epochs'][0]:
        if rank == 0:
            start_time = time.time()
            print('Epoch ', ep+1, ' of ', parameters['model_dict']['num_epochs'][0], ' lr_rate: ',lr_data[ep])
            sys.stdout.flush()

        if parameters['loader_dict']['shuffle_loader'] == True and ep > 0:  # reshuffle training data to avoid overfitting
            if ep % parameters['loader_dict']['shuffle_steps'] == 0:
                if rank == 0:
                    print('Shuffling training data...')
                if parameters['device_dict']['run_ddp']:
                    loader_train.sampler.set_epoch(ep)
                    loader_train = DataLoader(data['training'], batch_size=int(
                        parameters['loader_dict']['batch_size'][0] / parameters['device_dict']['world_size']),
                                              pin_memory=parameters['device_dict']['pin_memory'],follow_batch=follow_batch,
                                              sampler=DistributedSampler(data['training'],shuffle=parameters['loader_dict']['shuffle_loader'],
                                              seed=random.randint(-sys.maxsize - 1,sys.maxsize)),
                                              num_workers=parameters['loader_dict']['num_workers'])
                else:
                    loader_train = DataLoader(data['training'], pin_memory=parameters['device_dict']['pin_memory'],
                                              batch_size=parameters['loader_dict']['batch_size'][0], shuffle=parameters['loader_dict']['shuffle_loader'],
                                              follow_batch=follow_batch)


        parameters['model_dict']['optimizer_params']['params_group']['lr'] = lr_data[ep]
        if parameters['device_dict']['run_ddp']:
            parameters['model_dict']['optimizer_params']['params_group'][
                'params'] = model.module.processor.parameters()
        else:
            parameters['model_dict']['optimizer_params']['params_group']['params'] = model.processor.parameters()

        optimizer = set_optimizer(parameters)
        optimizer_to(optimizer, parameters['device_dict']['device'])

        loss_train = train(loader_train, model, parameters,optimizer,pretrain=True);
        if rank == 0:
            epoch_times.append(time.time() - start_time)
            print('epoch_time = ', time.time() - start_time, ' seconds Average epoch time = ',
                  sum(epoch_times) / float(len(epoch_times)), ' seconds')
            print('Train loss = ', loss_train)
            stats_file = open(os.path.join(parameters['io_dict']['pretrain_dir'], 'loss.data'), 'a')
            stats_file.write(str(loss_train) + '\n')
            stats_file.close()

        if ep > 0:
            delta_train = loss_train - L_train[-1]
            running_train_delta.append(abs(delta_train))
            if len(running_train_delta) > parameters['model_dict']['max_deltas']:
                running_train_delta.pop(0)
                if rank == 0:
                    print('Running training delta = ', sum(running_train_delta) / len(running_train_delta))

        L_train.append(loss_train)
        if loss_train < min_loss_train:
            min_loss_train = loss_train

            if rank == 0:
                model_name = glob.glob(os.path.join(parameters['io_dict']['pretrain_dir'], 'model_*'))
                if len(model_name) > 0:
                    os.remove(model_name[0])
                print('Saving model...')
                now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                if parameters['device_dict']['run_ddp']:
                    torch.save(model.module.state_dict(), os.path.join(parameters['io_dict']['pretrain_dir'], 'model_pre_'+str(now)))
                else:
                    torch.save(model.state_dict(), os.path.join(parameters['io_dict']['pretrain_dir'], 'model_pre_'+str(now)))
        if len(running_train_delta) == parameters['model_dict']['max_deltas']:
            if sum(running_train_delta) / len(running_train_delta) < parameters['model_dict']['train_tolerance']:
                if rank == 0:
                    print('Training delta satisfies set tolerance...exiting training loop...')
                ep = parameters['model_dict']['num_epochs'][0]
        ep += 1
    if parameters['device_dict']['run_ddp']:
        destroy_process_group()









