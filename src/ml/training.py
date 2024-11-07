import torch.distributed as dist
from numba import cuda
from torch import nn
import torch

from ..io.io import read_training_data, setup_model, setup_dataloader, save_model, save_dictionary
from .utils.distributed import ddp_destroy, ddp_setup, reduce_tensor
from ..utilities.distributions import get_distribution
from .utils.predict import accumulate_predictions
from .testing import test_non_intepretable
from .utils.optimizer import set_optimizer
from .utils.memory import optimizer_to

import numpy as np
import random
import shutil
import glob
import time
import sys
import os
import gc

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
    epoch_times = []
    running_valid_delta = []
    L_train, L_valid = [], []
    shuffle_counter = 0
    met_tolerance = 0
    min_loss_train = 1.0E30
    min_loss_valid = 1.0E30
    ep = 0
    parameters = ml.parameters
    if parameters['device_dict']['run_ddp']:
        ddp_setup(rank, parameters['device_dict']['world_size'], parameters['device_dict']['ddp_backend'])

    parameters['io_dict']['model_dir'] = None
    del parameters['io_dict']['model_dir']
    parameters['io_dict']['model_dir'] = os.path.join(parameters['io_dict']['main_path'],'models',
                                                      'training', str(iteration))
    if rank == 0:
        if os.path.isdir(parameters['io_dict']['model_dir']):
            shutil.rmtree(parameters['io_dict']['model_dir'])
        os.makedirs(parameters['io_dict']['model_dir'], exist_ok=True)
        print('Reading data...')

    data, samples = read_training_data(parameters,
                              os.path.join(parameters['io_dict']['samples_dir'], str(iteration), 'train_valid_split.npy'),format=parameters['io_dict']['graph_read_format'])
    model = setup_model(ml,rank=rank)
    loader_train, loader_valid = setup_dataloader(data=data,ml=ml)

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
    if rank == 0:
        print('Training model ',iteration,' using ',len(data['training']), ' training points and ',len(data['validation']),' validation points...')
    while ep < parameters['model_dict']['num_epochs'][1]:
        if rank == 0:
            start_time = time.time()
            print('Epoch ',ep+1,' of ',parameters['model_dict']['num_epochs'][1],  ' lr_rate: ',lr_data[ep], 'loss_accum: ',parameters['model_dict']['accumulate_loss'][1])
            sys.stdout.flush()

        if parameters['loader_dict']['shuffle_loader'] == True: #reshuffle training data to avoid overfitting
            if ep % parameters['loader_dict']['shuffle_steps'] == 0 and ep > 0:
                if rank == 0:
                    print('Shuffling training data...')
                loader_train, loader_valid = setup_dataloader(data=data,ml=ml,epoch=ep,reshuffle=True)
                shuffle_counter += 1

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

        L_train.append(loss_train)
        L_valid.append(loss_valid)
        if loss_train < min_loss_train:
            min_loss_train = loss_train
            if loss_valid < min_loss_valid:
                min_loss_valid = loss_valid
                if rank == 0:
                    model_params_group = {
                        'samples':samples,
                        'L_train':L_train[-1],
                        'L_valid':L_valid[-1]
                    }
                    save_model(model=model, ml=ml, model_params_group=model_params_group,remove_old_models=parameters['io_dict']['remove_old_model'])
        if ep > 0:
            delta_val = loss_valid - L_valid[-1]
            running_valid_delta.append(abs(delta_val))
            if len(running_valid_delta) > parameters['model_dict']['max_deltas']:
                running_valid_delta.pop(0)
                if rank == 0:
                    print('Running validation delta = ',sum(running_valid_delta)/len(running_valid_delta))
            if len(running_valid_delta) == parameters['model_dict']['max_deltas']:
                if sum(running_valid_delta)/len(running_valid_delta)< parameters['model_dict']['train_delta'][1] and (sum(L_valid[-parameters['model_dict']['max_deltas']:])/parameters['model_dict']['max_deltas']) < parameters['model_dict']['train_tolerance'][1]:
                    if rank == 0:
                        print('Validation delta satisfies set tolerance...exiting training loop...')
                    ep = parameters['model_dict']['num_epochs'][1]
                    met_tolerance = 1
        ep += 1
    if rank == 0:
        run_data = {
            'epoch_timings':epoch_times,
            'times_loader_shuffled':shuffle_counter,
            'met_tolerance':met_tolerance,
            'training_loss':L_train,
            'validation_loss': L_valid,
        }
        save_dictionary(fname=os.path.join(ml.parameters['io_dict']['model_dir'],'run_information.npy'),data=run_data)
    if parameters['device_dict']['run_ddp']:
        ddp_destroy()

def run_pre_training(rank,ml=None):
    epoch_times = []
    running_train_delta = []
    L_train = []
    shuffle_counter = 0
    met_tolerance = 0
    min_loss_train = 1.0E30
    ep = 0
    parameters = ml.parameters

    if parameters['device_dict']['run_ddp']:
        ddp_setup(rank, parameters['device_dict']['world_size'],parameters['device_dict']['ddp_backend'])

    parameters['io_dict']['model_dir'] = None
    del parameters['io_dict']['model_dir']
    parameters['io_dict']['model_dir'] = os.path.join(parameters['io_dict']['main_path'],'models','pretraining')
    if rank == 0:
        if os.path.isdir(parameters['io_dict']['model_dir']):
            shutil.rmtree(parameters['io_dict']['model_dir'])
        os.makedirs(parameters['io_dict']['model_dir'], exist_ok=True)
        print('Reading graphs...')

    data, samples = read_training_data(parameters,
                                       os.path.join(parameters['io_dict']['samples_dir'], 'train_valid_split.npy'),
                                       pretrain=True,format=parameters['io_dict']['graph_read_format'])
    model = setup_model(ml,rank=rank,pretrain=True)
    loader_train = setup_dataloader(data=data, ml=ml,pretrain=True)

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
    if rank == 0:
        print('Training using ', len(data['training']), ' training points')
    while ep < parameters['model_dict']['num_epochs'][0]:
        if rank == 0:
            start_time = time.time()
            print('Epoch ', ep+1, ' of ', parameters['model_dict']['num_epochs'][0], ' lr_rate: ',lr_data[ep])
            sys.stdout.flush()

        if parameters['loader_dict']['shuffle_loader'] == True and ep > 0:  # reshuffle training data to avoid overfitting
            if ep % parameters['loader_dict']['shuffle_steps'] == 0:
                if rank == 0:
                    print('Shuffling training data...')
                loader_train = setup_dataloader(data=data, ml=ml,epoch=ep,pretrain=True,reshuffle=True)
                shuffle_counter += 1

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

        L_train.append(loss_train)
        if loss_train < min_loss_train:
            min_loss_train = loss_train
            if rank == 0:
                model_params_group = {
                    'samples': samples,
                    'L_train': L_train[-1],
                }
                save_model(model=model, ml=ml, model_params_group=model_params_group,pretrain=True)
        if ep > 0:
            delta_train = loss_train - L_train[-1]
            running_train_delta.append(abs(delta_train))
            if len(running_train_delta) > parameters['model_dict']['max_deltas']:
                running_train_delta.pop(0)
                if rank == 0:
                    print('Running training delta = ', sum(running_train_delta) / len(running_train_delta))
            if len(running_train_delta) == parameters['model_dict']['max_deltas']:
                if sum(running_train_delta) / len(running_train_delta) < parameters['model_dict']['train_delta'][0] and (sum(L_train[-parameters['model_dict']['max_deltas']:])/parameters['model_dict']['max_deltas']) < parameters['model_dict']['train_tolerance'][0]:
                    if rank == 0:
                        print('Training delta satisfies set tolerance...exiting training loop...')
                    ep = parameters['model_dict']['num_epochs'][0]
                    met_tolerance = 1
        ep += 1
    if rank == 0:
        run_data = {
            'epoch_timings':epoch_times,
            'times_loader_shuffled':shuffle_counter,
            'met_tolerance':met_tolerance,
            'training_loss': L_train
        }
        save_dictionary(fname=os.path.join(ml.parameters['io_dict']['model_dir'],'run_information.npy'),data=run_data)
    if parameters['device_dict']['run_ddp']:
        ddp_destroy()









