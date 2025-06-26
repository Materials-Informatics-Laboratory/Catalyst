import torch.distributed as dist
from numba import cuda
from torch import nn
import torch

from ..io.data_management import read_training_data, setup_model, setup_dataloader, save_model, save_dictionary, read_graphs_from_gids
from .utils.distributed import ddp_destroy, ddp_setup, reduce_tensor
from ..utilities.distributions import get_distribution
from .utils.predict import accumulate_predictions
from .inference import test_non_intepretable_internal
from ..utilities.sampling import active_sampling, random_
from .utils.optimizer import set_optimizer
from .utils.memory import optimizer_to
from .utils.loss import loss_setup,active_learning_setup


import numpy as np
import random
import shutil
import glob
import time
import sys
import os
import gc

def train(loader,model,parameters,optimizer,active_learning_dict=None):
    model.train()
    loss_accum = parameters['model_dict']['accumulate_loss']
    epoch_loss = 0.0
    if parameters['device_dict']['run_ddp'] == False:
        model.to(parameters['device_dict']['device'])
    loss_fn = loss_setup(params=parameters['model_dict']['loss_params'])
    for data in loader:
        def closure():
            data.to(parameters['device_dict']['device'], non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(data)
            preds, y, vec = accumulate_predictions(pred,data,loss_accum)
            preds = preds.to(y.device)
            if vec:
                loss_list = [0.0]*len(preds)
                for i in range(len(preds)):
                    loss_list[i] = loss_fn(preds[i],y[i])
                batch_loss = torch.sum(torch.stack(loss_list))
            else:
                batch_loss = loss_fn(preds,y)
            nonlocal epoch_loss
            if active_learning_dict is not None:
                batch_loss = active_learning_dict['loss_regularization'].get_loss(model, batch_loss,
                                                                                  active_learning_dict['lambda'],
                                                                                  )
            epoch_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            return batch_loss
        optimizer.step(closure)
    if parameters['device_dict']['run_ddp']:
        epoch_loss = reduce_tensor(torch.tensor(epoch_loss).to(parameters['device_dict']['device'])).item()
    return epoch_loss / (len(loader)*parameters['device_dict']['world_size'])

def run_active_learning(rank,cat=None):
    # set up run
    epoch_times = []
    running_valid_delta = []
    L_train, L_valid = [], []
    shuffle_counter = 0
    met_tolerance = 0
    min_loss_train = 1.0E30
    min_loss_valid = 1.0E30
    iteration = 0
    parameters = cat.parameters

    if parameters['device_dict']['run_ddp']:
        ddp_setup(rank, parameters['device_dict']['world_size'], parameters['device_dict']['ddp_backend'])

    parameters['io_dict']['model_dir'] = None
    del parameters['io_dict']['model_dir']
    parameters['io_dict']['model_dir'] = os.path.join(parameters['io_dict']['main_path'], 'models','active_learning')
    if rank == 0:
        if os.path.isdir(parameters['io_dict']['model_dir']):
            shutil.rmtree(parameters['io_dict']['model_dir'])
        os.makedirs(parameters['io_dict']['model_dir'], exist_ok=True)

    model, model_data = setup_model(cat, rank=rank, load=True)
    active_loss_dict = active_learning_setup(parameters,{'model':model,'metadata':model_data})

    if parameters['io_dict']['graph_read_format'] != 2:
        training_graphs = read_graphs_from_gids(parameters['model_dict']['active_learning_params_group']['training_data_dir'],model_data['samples']['training_samples'])
    else:
        training_graphs = load_dictionary(glob.glob(os.path.join(parameters['model_dict']['active_learning_params_group']['training_data_dir'], 'graphs.data'))[0])['graphs']
    new_graphs = read_graphs_from_gids(
        parameters['io_dict']['data_dir'])

    parameters['model_dict']['active_learning_params_group']['sampling_params_group']['data'] = new_graphs
    parameters['model_dict']['active_learning_params_group']['sampling_params_group']['training_data'] = training_graphs
    parameters['model_dict']['active_learning_params_group']['sampling_params_group']['y'] = [graph.y for graph in new_graphs]
    parameters['model_dict']['active_learning_params_group']['sampling_params_group']['training_y'] = [graph.y for graph in training_graphs]

    # retrain model for a few epochs with new data added
    data = {
        'training': [],
        'validation': []
    }
    if parameters['model_dict']['active_learning_params_group']['training_params_group']['train_with_previous']:
        if isinstance(parameters['model_dict']['active_learning_params_group']['training_params_group']['percent_use_previous'],float):
            samples, non_samples = random_(training_graphs.copy(),
                                          parameters['model_dict']['active_learning_params_group']['training_params_group']['percent_use_previous'],
                                          rng=parameters['model_dict']['active_learning_params_group']['sampling_params_group']['rng'])
            data['training'] = [training_graphs[i] for i in samples]
        else:
            data['training'] = training_graphs.copy()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, parameters['model_dict']['active_learning_params_group']['training_params_group']['iterations'] + 1, sharex=False,sharey=False)
    loader_train = setup_dataloader({
        'training': training_graphs,
    }, cat=cat, mode=0)
    loss_train, train_info = test_non_intepretable_internal(loader_train, model, parameters, rank=rank,
                                                            return_test_info=True)
    ax[2][0].plot([item for sublist in train_info['y'] for item in sublist],
                  [item for sublist in train_info['pred'] for item in sublist], 'yo', markeredgecolor='k')
    while iteration < parameters['model_dict']['active_learning_params_group']['training_params_group']['iterations']:
        # determine which point(s) to add to model
        new_samples, remaining_data = active_sampling(
            parameters['model_dict']['active_learning_params_group']['sampling_params_group'])
        loader_train = setup_dataloader({'training': new_graphs, }, cat=cat, mode=0)
        loss_train, train_info = test_non_intepretable_internal(loader_train, model, parameters, rank=rank,
                                                               return_test_info=True)
        ty = np.linspace(0,50,10)
        for i in range(parameters['model_dict']['active_learning_params_group']['training_params_group']['iterations'] + 1):
            ax[0][i].plot(ty,ty,'k')
            ax[1][i].plot(ty, ty, 'k')
            ax[2][i].plot(ty, ty, 'k')
        ax[0][0].plot([item for sublist in train_info['y'] for item in sublist],
                [item for sublist in train_info['pred'] for item in sublist],'co',markeredgecolor='k')

        data['training'].extend([new_graphs[i] for i in new_samples])
        data['validation'] = [new_graphs[i] for i in remaining_data]
        # add points to dataloader for model
        loader_train, loader_valid = setup_dataloader(data=data, cat=cat, mode=1)

        ep = 0
        if parameters['model_dict']['optimizer_params']['dynamic_lr']:
            dist_params = dict(
                dist_type=parameters['model_dict']['optimizer_params']['dist_type'],
                vars=parameters['model_dict']['optimizer_params']['lr_scale'],
                size=parameters['model_dict']['num_epochs'],
                floor=parameters['model_dict']['optimizer_params']['params_group']['lr']
            )
            lr_data = get_distribution(dist_params)
        else:
            lr_data = np.linspace(parameters['model_dict']['optimizer_params']['params_group']['lr'],
                                  parameters['model_dict']['optimizer_params']['params_group']['lr'],
                                  parameters['model_dict']['active_learning_params_group']['training_params_group']['epochs_per_iteration'])

        while ep < parameters['model_dict']['active_learning_params_group']['training_params_group']['epochs_per_iteration']:
            parameters['model_dict']['optimizer_params']['params_group']['lr'] = lr_data[ep]
            if parameters['device_dict']['run_ddp']:
                parameters['model_dict']['optimizer_params']['params_group'][
                    'params'] = model.module.processor.parameters()
            else:
                parameters['model_dict']['optimizer_params']['params_group']['params'] = model.processor.parameters()
            optimizer = set_optimizer(parameters)
            optimizer_to(optimizer, parameters['device_dict']['device'])
            loss_train = train(loader_train, model, parameters, optimizer,active_learning_dict=active_loss_dict);
            ep += 1

        loss_valid, test_info = test_non_intepretable_internal(loader_valid, model, parameters, rank=rank,
                                                               return_test_info=True)
        loss_train, train_info = test_non_intepretable_internal(loader_train, model, parameters, rank=rank,
                                                                return_test_info=True)
        ax[0][iteration + 1].plot([item for sublist in test_info['y'] for item in sublist],
                        [item for sublist in test_info['pred'] for item in sublist], 'go',markeredgecolor='k')
        ax[1][iteration + 1].plot([item for sublist in train_info['y'] for item in sublist],
                        [item for sublist in train_info['pred'] for item in sublist], 'ro',markeredgecolor='k')

        new_graphs = [new_graphs[i] for i in remaining_data]
        parameters['model_dict']['active_learning_params_group']['sampling_params_group']['data'] = new_graphs
        parameters['model_dict']['active_learning_params_group']['sampling_params_group']['y'] = [graph.y for graph in
                                                                                                  new_graphs]

        loader_train = setup_dataloader({
            'training': training_graphs,
        }, cat=cat, mode=0)
        loss_train, train_info = test_non_intepretable_internal(loader_train, model, parameters, rank=rank,
                                                                return_test_info=True)
        ax[2][iteration+1].plot([item for sublist in train_info['y'] for item in sublist],
                      [item for sublist in train_info['pred'] for item in sublist], 'yo', markeredgecolor='k')

        iteration += 1
    plt.show()

    # test model against added points

    # determine if desired threshold has been met, if not, loop, if yes, exit


def run_training(rank,iteration,cat=None):
    epoch_times = []
    running_valid_delta = []
    L_train, L_valid = [], []
    shuffle_counter = 0
    met_tolerance = 0
    min_loss_train = 1.0E30
    min_loss_valid = 1.0E30
    ep = 0
    parameters = cat.parameters
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
                              os.path.join(parameters['io_dict']['samples_dir'], str(iteration), 'train_valid_split.npy'),
                                       format=parameters['io_dict']['graph_read_format'],rank=rank)
    load_model = False
    if parameters['model_dict']['pre_training'] or parameters['model_dict']['restart_training']:
        load_model=True
    model = setup_model(cat,rank=rank,load=load_model)
    loader_train, loader_valid = setup_dataloader(data=data,cat=cat,mode=1)

    if parameters['model_dict']['optimizer_params']['dynamic_lr']:
        dist_params = dict(
                dist_type=parameters['model_dict']['optimizer_params']['dist_type'],
                vars=parameters['model_dict']['optimizer_params']['lr_scale'],
                size=parameters['model_dict']['num_epochs'],
                floor=parameters['model_dict']['optimizer_params']['params_group']['lr']
            )
        lr_data = get_distribution(dist_params)
    else:
        lr_data = np.linspace(parameters['model_dict']['optimizer_params']['params_group']['lr'],
                                  parameters['model_dict']['optimizer_params']['params_group']['lr'],parameters['model_dict']['num_epochs'])
    if rank == 0:
        print('Training model ',iteration,' using ',len(data['training']), ' training points and ',len(data['validation']),' validation points...')
    while ep < parameters['model_dict']['num_epochs']:
        if rank == 0:
            if ep > 0:
                start_time = time.time()
            print('Epoch ',ep+1,' of ',parameters['model_dict']['num_epochs'],  ' lr_rate: ',lr_data[ep], 'loss_accum: ',parameters['model_dict']['accumulate_loss'])
            sys.stdout.flush()

        if parameters['loader_dict']['shuffle_loader'] == True: #reshuffle training data to avoid overfitting
            if ep % parameters['loader_dict']['shuffle_steps'] == 0 and ep > 0:
                if rank == 0:
                    print('Shuffling training data...')
                loader_train, loader_valid = setup_dataloader(data=data,cat=cat,epoch=ep,reshuffle=True,mode=1)
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
        loss_valid = test_non_intepretable_internal(loader_valid, model, parameters,rank=rank)

        if rank == 0:
            if ep > 0:
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
                        'data_loader':loader_train,
                        'L_train':L_train[-1],
                        'L_valid':L_valid[-1]
                    }
                    save_model(model=model, cat=cat, model_params_group=model_params_group,remove_old_models=parameters['io_dict']['remove_old_model'])
        if ep > 1:
            delta_val = loss_valid - L_valid[-2]
            running_valid_delta.append(abs(delta_val))
            if len(running_valid_delta) > parameters['model_dict']['max_deltas']:
                running_valid_delta.pop(0)
                if rank == 0:
                    print('Running validation delta = ',sum(running_valid_delta)/len(running_valid_delta))
            if len(running_valid_delta) == parameters['model_dict']['max_deltas']:
                if sum(running_valid_delta)/len(running_valid_delta)< parameters['model_dict']['train_delta'] and (sum(L_valid[-parameters['model_dict']['max_deltas']:])/parameters['model_dict']['max_deltas']) < parameters['model_dict']['train_tolerance']:
                    if rank == 0:
                        print('Validation delta satisfies set tolerance...exiting training loop...')
                    ep = parameters['model_dict']['num_epochs']
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
        save_dictionary(fname=os.path.join(cat.parameters['io_dict']['model_dir'],'run_information.npy'),data=run_data)
    if parameters['device_dict']['run_ddp']:
        ddp_destroy()

def run_pre_training(rank,cat=None):
    epoch_times = []
    running_train_delta = []
    L_train = []
    shuffle_counter = 0
    met_tolerance = 0
    min_loss_train = 1.0E30
    ep = 0
    parameters = cat.parameters

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
                                       pretrain=True,format=parameters['io_dict']['graph_read_format'],rank=rank)
    model = setup_model(cat,rank=rank)
    loader_train = setup_dataloader(data=data, cat=cat,mode=0)

    if parameters['model_dict']['optimizer_params']['dynamic_lr']:
        dist_params = dict(
            dist_type=parameters['model_dict']['optimizer_params']['dist_type'],
            vars=parameters['model_dict']['optimizer_params']['lr_scale'],
            size=parameters['model_dict']['num_epochs'],
            floor=parameters['model_dict']['optimizer_params']['params_group']['lr']
        )
        lr_data = get_distribution(dist_params)
    else:
        lr_data = np.linspace(parameters['model_dict']['optimizer_params']['params_group']['lr'],
                              parameters['model_dict']['optimizer_params']['params_group']['lr'],
                              parameters['model_dict']['num_epochs'])
    if rank == 0:
        print('Training using ', len(data['training']), ' training points')
    while ep < parameters['model_dict']['num_epochs']:
        if rank == 0:
            if ep > 0:
                start_time = time.time()
            print('Epoch ', ep+1, ' of ', parameters['model_dict']['num_epochs'], ' lr_rate: ',lr_data[ep])
            sys.stdout.flush()

        if parameters['loader_dict']['shuffle_loader'] == True and ep > 0:  # reshuffle training data to avoid overfitting
            if ep % parameters['loader_dict']['shuffle_steps'] == 0:
                if rank == 0:
                    print('Shuffling training data...')
                loader_train = setup_dataloader(data=data, cat=cat,epoch=ep,reshuffle=True,mode=0)
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
            if ep > 0:
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
                save_model(model=model, cat=cat, model_params_group=model_params_group,pretrain=True)
        if ep > 1:
            delta_train = loss_train - L_train[-2]
            running_train_delta.append(abs(delta_train))
            if len(running_train_delta) > parameters['model_dict']['max_deltas']:
                running_train_delta.pop(0)
                if rank == 0:
                    print('Running training delta = ', sum(running_train_delta) / len(running_train_delta))
            if len(running_train_delta) == parameters['model_dict']['max_deltas']:
                if sum(running_train_delta) / len(running_train_delta) < parameters['model_dict']['train_delta'] and (sum(L_train[-parameters['model_dict']['max_deltas']:])/parameters['model_dict']['max_deltas']) < parameters['model_dict']['train_tolerance']:
                    if rank == 0:
                        print('Training delta satisfies set tolerance...exiting training loop...')
                    ep = parameters['model_dict']['num_epochs']
                    met_tolerance = 1
        ep += 1
    if rank == 0:
        run_data = {
            'epoch_timings':epoch_times,
            'times_loader_shuffled':shuffle_counter,
            'met_tolerance':met_tolerance,
            'training_loss': L_train
        }
        save_dictionary(fname=os.path.join(cat.parameters['io_dict']['model_dir'],'run_information.npy'),data=run_data)
    if parameters['device_dict']['run_ddp']:
        ddp_destroy()









