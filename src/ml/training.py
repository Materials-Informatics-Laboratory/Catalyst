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

from datetime import datetime
import numpy as np
import random
import shutil
import glob
import time
import sys
import os
import gc

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
def ddp_setup(rank: int,world_size,backend):
    """
    Args:
    rank: Unique identifier of each process
    world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    
    init_process_group(backend=backend, rank=rank, world_size=world_size)

def ddp_destroy():
    dist.barrier()
    destroy_process_group()

def train(loader,model,parameters,optimizer,pretrain=False):
    model.train()

    if pretrain:
        loss_accum = parameters['model_dict']['accumulate_loss'][0]
    else:
        loss_accum = parameters['model_dict']['accumulate_loss'][1]

    total_loss = 0.0

    if parameters['run_ddp'] == False:
        model.to(parameters['device'])
    loss_fn = parameters['model_dict']['loss_func']
    for data in loader:
        def closure():
            data.to(parameters['device'], non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            pred = model(data)

            if loss_accum == 'exact':
                if hasattr(data, 'x_atm_batch'):
                    d = pred[0].flatten()
                    x = torch.unique(data['x_atm_batch'])
                    sorted_atms = [None] * len(x)
                    dd = data['x_atm_batch']
                    for i,xx in enumerate(x):
                        xw = torch.where(dd == xx)
                        sorted_atms[i] = d[xw]
                    d = pred[1].flatten()
                    x = torch.unique(data['x_bnd_batch'])
                    sorted_bnds = [None] * len(x)
                    dd = data['x_bnd_batch']
                    for i,xx in enumerate(x):
                        xw = torch.where(dd == xx)
                        sorted_bnds[i] = d[xw]
                    if hasattr(data,'x_ang_batch'):
                        d = pred[2].flatten()
                        x = torch.unique(data['x_ang_batch'])
                        sorted_angs = [None] * len(x)
                        dd = data['x_ang_batch']
                        for i,xx in enumerate(x):
                            xw = torch.where(dd == xx)
                            sorted_angs[i] = d[xw]

                    preds = []
                    for i in range(len(sorted_atms)):
                        if hasattr(data, 'x_ang_batch'):
                            preds.append(sorted_atms[i].sum() + sorted_angs[i].sum() + sorted_bnds[i].sum())
                        else:
                            preds.append(sorted_atms[i].sum() + sorted_bnds[i].sum())
                    preds = torch.stack(preds)

                    y = data.y.flatten()
                else:
                    '''
                    Implement generic routines here for non alignn systems
                    '''
                    pass
            elif loss_accum == 'sum':
                if hasattr(data, 'x_ang_batch') and hasattr(data, 'x_ang_batch'):
                    preds = pred[0].sum() + pred[1].sum() + pred[2].sum()
                else:
                    preds = pred[0].sum() + pred[1].sum()
                y = data.y.flatten().sum()

            loss = loss_fn(preds,y)
            nonlocal total_loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            return loss
        optimizer.step(closure)
        #if parameters['run_ddp']:
        #    data.detach()
    return total_loss/(len(loader)*parameters['world_size'])

def run_training(rank,ml=None):

    parameters = ml.parameters
    for iteration in range(parameters['model_dict']['n_models']):
        parameters['model_save_dir'] = os.path.join(parameters['model_dir'], str(iteration))
        if rank == 0:
            if os.path.isdir(parameters['model_save_dir']):
                shutil.rmtree(parameters['model_save_dir'])
            os.mkdir(parameters['model_save_dir'])
        ml.set_model()
        model = ml.model
        if parameters['restart_training']:
            print('Restarting model training using model ', parameters['restart_model_name'])

        partitioned_data = np.load(
            os.path.join(parameters['graph_data_dir'], 'model_samples', str(iteration), 'train_valid_split.npy'),
            allow_pickle=True)

        data = dict(training=partitioned_data.item().get('training'),
                       validation=partitioned_data.item().get('validation'))

        if rank == 0:
            print('Training model ', iteration)
            print('Training using ',len(data['training']), ' training points and ',len(data['validation']),' validation points...')
            stats_file = open(os.path.join(parameters['model_save_dir'], 'loss.data'), 'w')
            stats_file.write('Training_loss     Validation loss\n')
            stats_file.close()

        if parameters['run_ddp']:
            ddp_setup(rank,parameters['world_size'],parameters['ddp_backend'])
        if parameters['pre_training'] == False:
            if parameters['restart_training']:
                device = torch.device(ml.parameters['device'])
                ml.model.load_state_dict(torch.load(ml.parameters['restart_model_name'],map_location=device))
            model.to(parameters['device'])
            if parameters['run_ddp']:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = DDP(model, device_ids=[rank],find_unused_parameters=True)
        else:
            model.to(ml.parameters['device'])
            device = torch.device(ml.parameters['device'])
            model_name = glob.glob(os.path.join(parameters['pretrain_dir'], 'model_*'))
            model.load_state_dict(torch.load(model_name[0],map_location=device))
            if parameters['run_ddp']:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(data['training'][0], 'x_ang') else ['x_atm','x_bnd']
        if parameters['run_ddp']:
            if parameters['loader_dict']['shuffle_loader'] == False:
                loader_train = DataLoader(data['training'], batch_size=int(
                    parameters['loader_dict']['batch_size'][1] / parameters['world_size']),
                                      pin_memory=parameters['pin_memory'],
                                      follow_batch=follow_batch,
                                      sampler=DistributedSampler(data['training'],
                                                                 shuffle=False),num_workers=parameters['loader_dict']['num_workers'])
            loader_valid = DataLoader(data['validation'], follow_batch=follow_batch,batch_size=int(
                        parameters['loader_dict']['batch_size'][2] / parameters['world_size']),pin_memory=parameters['pin_memory'],
                                      shuffle=False, sampler=DistributedSampler(data['validation']),num_workers=parameters['loader_dict']['num_workers'])
        else:
            loader_train = DataLoader(data['training'], pin_memory=parameters['pin_memory'],
                                      batch_size=parameters['loader_dict']['batch_size'][1],
                                      shuffle=parameters['loader_dict']['shuffle_loader'],
                                      follow_batch=follow_batch,
                                      num_workers=parameters['loader_dict']['num_workers'])

            loader_valid = DataLoader(data['validation'],follow_batch=follow_batch,pin_memory=parameters['pin_memory'],
                                      batch_size=int(
                        parameters['loader_dict']['batch_size'][2] / parameters['world_size']), shuffle=False,num_workers=parameters['loader_dict']['num_workers'])

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
        for ep in range(parameters['model_dict']['num_epochs'][1]):
            if rank == 0:
                start_time = time.time()
                print('Epoch ',ep+1,' of ',parameters['model_dict']['num_epochs'][1],  ' lr_rate: ',lr_data[ep], 'loss_accum: ',parameters['model_dict']['accumulate_loss'])
                sys.stdout.flush()

            if parameters['loader_dict']['shuffle_loader'] == True: #reshuffle training data to avoid overfitting
                if rank == 0:
                    print('Shuffling training data...')
                if parameters['run_ddp']:
                    if ep > 0:
                        loader_train.sampler.set_epoch(ep)
                        loader_valid.sampler.set_epoch(ep)
                    loader_train = DataLoader(data['training'], batch_size=int(
                        parameters['loader_dict']['batch_size'][1] / parameters['world_size']),
                                              pin_memory=parameters['pin_memory'],
                                              follow_batch=follow_batch,
                                              sampler=DistributedSampler(data['training'],shuffle=parameters['loader_dict']['shuffle_loader'],
                                                                         seed=random.randint(-sys.maxsize - 1, sys.maxsize)),
                                              num_workers=parameters['loader_dict']['num_workers'])

            parameters['model_dict']['optimizer_params']['params_group']['lr'] = lr_data[ep]
            if parameters['run_ddp']:
                parameters['model_dict']['optimizer_params']['params_group'][
                    'params'] = model.module.processor.parameters()
            else:
                parameters['model_dict']['optimizer_params']['params_group']['params'] = model.processor.parameters()

            if parameters['model_dict']['optimizer_params']['optimizer'] == 'AdamW':
                optimizer = torch.optim.AdamW([parameters['model_dict']['optimizer_params']['params_group']])
            elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Adadelta':
                optimizer = torch.optim.Adadelta([parameters['model_dict']['optimizer_params']['params_group']])
            elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Adagrad':
                optimizer = torch.optim.Adagrad([parameters['model_dict']['optimizer_params']['params_group']])
            elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Adam':
                optimizer = torch.optim.Adam([parameters['model_dict']['optimizer_params']['params_group']])
            elif parameters['model_dict']['optimizer_params']['optimizer'] == 'SparseAdam':
                optimizer = torch.optim.SparseAdam([parameters['model_dict']['optimizer_params']['params_group']])
            elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Adamax':
                optimizer = torch.optim.Adamax([parameters['model_dict']['optimizer_params']['params_group']])
            elif parameters['model_dict']['optimizer_params']['optimizer'] == 'ASGD':
                optimizer = torch.optim.ASGD([parameters['model_dict']['optimizer_params']['params_group']])
            elif parameters['model_dict']['optimizer_params']['optimizer'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([parameters['model_dict']['optimizer_params']['params_group']])
            elif parameters['model_dict']['optimizer_params']['optimizer'] == 'NAdam':
                optimizer = torch.optim.NAdam([parameters['model_dict']['optimizer_params']['params_group']])
            elif parameters['model_dict']['optimizer_params']['optimizer'] == 'RAdam':
                optimizer = torch.optim.RAdam([parameters['model_dict']['optimizer_params']['params_group']])
            elif parameters['model_dict']['optimizer_params']['optimizer'] == 'RMSprop':
                optimizer = torch.optim.RMSprop([parameters['model_dict']['optimizer_params']['params_group']])
            elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Rprop':
                optimizer = torch.optim.Rprop([parameters['model_dict']['optimizer_params']['params_group']])
            elif parameters['model_dict']['optimizer_params']['optimizer'] == 'SGD':
                optimizer = torch.optim.SGD([parameters['model_dict']['optimizer_params']['params_group']])
            optimizer_to(optimizer,parameters['device'])

            loss_train = train(loader_train, model, parameters, optimizer);
            loss_valid = test_non_intepretable(loader_valid, model, parameters)

            if rank == 0:
                epoch_times.append(time.time() - start_time)
                print('epoch_time = ', time.time() - start_time, ' seconds Average epoch time = ', sum(epoch_times) / float(len(epoch_times)), ' seconds')
                print('Train loss = ',loss_train,' Validation loss = ',loss_valid)
                if ep > 0:
                    delta_val = loss_valid - L_valid[-1]
                    running_valid_delta.append(abs(delta_val))
                    if len(running_valid_delta) > 3:
                        running_valid_delta.pop(0)
                    print('Running validation delta = ',sum(running_valid_delta)/len(running_valid_delta))
                stats_file = open(os.path.join(parameters['model_save_dir'], 'loss.data'), 'a')
                stats_file.write(str(loss_train) + '     ' + str(loss_valid) + '\n')
                stats_file.close()
            L_train.append(loss_train)
            L_valid.append(loss_valid)

            if loss_train < min_loss_train:
                min_loss_train = loss_train
                if loss_valid < min_loss_valid:
                    min_loss_valid = loss_valid
                    if rank == 0:
                        if parameters['remove_old_model']:
                            model_name = glob.glob(os.path.join(parameters['model_save_dir'], 'model_*'))
                            if len(model_name) > 0 and rank == 0:
                                os.remove(model_name[0])
                        print('Saving model...')
                        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                        if parameters['run_ddp']:
                            torch.save(model.module.state_dict(), os.path.join(parameters['model_save_dir'], 'model_' + str(now)+'_ep-'+str(ep)))
                        else:
                            torch.save(model.state_dict(), os.path.join(parameters['model_save_dir'], 'model_' + str(now)+'_ep-'+str(ep)))
            if len(running_valid_delta) > 2:
                if sum(running_valid_delta)/len(running_valid_delta)< parameters['model_dict']['train_tolerance']:
                    if rank == 0:
                        if parameters['run_ddp']:
                            ddp_destroy()
                        print('Validation delta satisfies set tolerance...exiting training loop...')
                    break
        if parameters['run_ddp']:
            ddp_destroy()

def run_pre_training(rank,ml=None):
    parameters = ml.parameters
    if rank == 0:
        if os.path.isdir(parameters['pretrain_dir']):
            shutil.rmtree(parameters['pretrain_dir'])
        os.mkdir(parameters['pretrain_dir'])

    model = ml.model
    partitioned_data = np.load(os.path.join(parameters['graph_data_dir'], 'pre_training', 'train_valid_split.npy'),
                               allow_pickle=True)
    data = dict(training=partitioned_data.item().get('training'), validation=None)
    if rank == 0:
        print('Training using ', len(data['training']), ' training points')
        stats_file = open(os.path.join(parameters['pretrain_dir'], 'loss.data'), 'w')
        stats_file.write('Training_loss\n')
        stats_file.close()

    if parameters['run_ddp']:
        ddp_setup(rank, parameters['world_size'],parameters['ddp_backend'])
        model.to(parameters['device'])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank],find_unused_parameters=True)

    follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(data['training'][0], 'x_ang') else ['x_atm']
    if parameters['run_ddp']:
        loader_train = DataLoader(data['training'], batch_size=parameters['loader_dict']['batch_size'][0],
                                  pin_memory=parameters['pin_memory'],
                                  shuffle=False, follow_batch=follow_batch,
                                  sampler=DistributedSampler(data['training']))
    else:
        loader_train = DataLoader(data['training'], pin_memory=parameters['pin_memory'],
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

        if parameters['loader_dict']['shuffle_loader'] == True:  # reshuffle training data to avoid overfitting
            if rank == 0:
                print('Shuffling training data...')
            if parameters['run_ddp']:
                if ep > 0:
                    loader_train.sampler.set_epoch(ep)
                loader_train = DataLoader(data['training'], batch_size=int(
                    parameters['loader_dict']['batch_size'][0] / parameters['world_size']),
                                          pin_memory=parameters['pin_memory'],follow_batch=follow_batch,
                                          sampler=DistributedSampler(data['training'],shuffle=parameters['loader_dict']['shuffle_loader'],
                                          seed=random.randint(-sys.maxsize - 1,sys.maxsize)),
                                          num_workers=parameters['loader_dict']['num_workers'])

        parameters['model_dict']['optimizer_params']['params_group']['lr'] = lr_data[ep]
        if parameters['run_ddp']:
            parameters['model_dict']['optimizer_params']['params_group'][
                'params'] = model.module.processor.parameters()
        else:
            parameters['model_dict']['optimizer_params']['params_group']['params'] = model.processor.parameters()

        if parameters['model_dict']['optimizer_params']['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW([parameters['model_dict']['optimizer_params']['params_group']])
        elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Adadelta':
            optimizer = torch.optim.Adadelta([parameters['model_dict']['optimizer_params']['params_group']])
        elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Adagrad':
            optimizer = torch.optim.Adagrad([parameters['model_dict']['optimizer_params']['params_group']])
        elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam([parameters['model_dict']['optimizer_params']['params_group']])
        elif parameters['model_dict']['optimizer_params']['optimizer'] == 'SparseAdam':
            optimizer = torch.optim.SparseAdam([parameters['model_dict']['optimizer_params']['params_group']])
        elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Adamax':
            optimizer = torch.optim.Adamax([parameters['model_dict']['optimizer_params']['params_group']])
        elif parameters['model_dict']['optimizer_params']['optimizer'] == 'ASGD':
            optimizer = torch.optim.ASGD([parameters['model_dict']['optimizer_params']['params_group']])
        elif parameters['model_dict']['optimizer_params']['optimizer'] == 'LBFGS':
            optimizer = torch.optim.LBFGS([parameters['model_dict']['optimizer_params']['params_group']])
        elif parameters['model_dict']['optimizer_params']['optimizer'] == 'NAdam':
            optimizer = torch.optim.NAdam([parameters['model_dict']['optimizer_params']['params_group']])
        elif parameters['model_dict']['optimizer_params']['optimizer'] == 'RAdam':
            optimizer = torch.optim.RAdam([parameters['model_dict']['optimizer_params']['params_group']])
        elif parameters['model_dict']['optimizer_params']['optimizer'] == 'RMSprop':
            optimizer = torch.optim.RMSprop([parameters['model_dict']['optimizer_params']['params_group']])
        elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Rprop':
            optimizer = torch.optim.Rprop([parameters['model_dict']['optimizer_params']['params_group']])
        elif parameters['model_dict']['optimizer_params']['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD([parameters['model_dict']['optimizer_params']['params_group']])
        optimizer_to(optimizer, parameters['device'])

        loss_train = train(loader_train, model, parameters,optimizer,pretrain=True);
        if rank == 0:
            epoch_times.append(time.time() - start_time)
            print('epoch_time = ', time.time() - start_time, ' seconds Average epoch time = ',
                  sum(epoch_times) / float(len(epoch_times)), ' seconds')
            print('Train loss = ', loss_train)

        if ep > 0:
            delta_train = loss_train - L_train[-1]
            running_train_delta.append(abs(delta_train))
            if len(running_train_delta) > 3:
                running_train_delta.pop(0)
                if rank == 0:
                    print('Running training delta = ', sum(running_train_delta) / len(running_train_delta))

            stats_file = open(os.path.join(parameters['pretrain_dir'], 'loss.data'), 'a')
            stats_file.write(str(loss_train) + '\n')
            stats_file.close()
        L_train.append(loss_train)
        if loss_train < min_loss_train:
            min_loss_train = loss_train

            if rank == 0:
                model_name = glob.glob(os.path.join(parameters['pretrain_dir'], 'model_*'))
                if len(model_name) > 0:
                    os.remove(model_name[0])
                print('Saving model...')
                now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                if parameters['run_ddp']:
                    torch.save(model.module.state_dict(), os.path.join(parameters['pretrain_dir'], 'model_pre_'+str(now)))
                else:
                    torch.save(model.state_dict(), os.path.join(parameters['pretrain_dir'], 'model_pre_'+str(now)))
        if len(running_train_delta) > 2:
            if sum(running_train_delta) / len(running_train_delta) < parameters['model_dict']['train_tolerance']:
                if rank == 0:
                    print('Training delta satisfies set tolerance...exiting training loop...')
                ep = parameters['model_dict']['num_epochs'][0]
        ep += 1
    if parameters['run_ddp']:
        gc.collect()
        torch.cuda.empty_cache()
        destroy_process_group()









