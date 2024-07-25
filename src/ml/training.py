from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group,destroy_process_group
from torch_geometric.loader import DataLoader
from torch import nn
import torch

from .testing import test_intepretable, test_non_intepretable

from datetime import datetime
import glob
import sys
import os

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
    destroy_process_group()

def train(loader,model,parameters,PIN_MEMORY=False):
    model.train()
    total_loss = 0.0
    if parameters['run_ddp']:
        optimizer = torch.optim.AdamW(model.module.processor.parameters(), lr=parameters['LEARN_RATE'])
    else:
        optimizer = torch.optim.AdamW(model.processor.parameters(), lr=parameters['LEARN_RATE'])
        model.to(parameters['device'])
    loss_fn = torch.nn.MSELoss()
    for i,data in enumerate(loader, 0):
        data.to(parameters['device'])
        optimizer.zero_grad(set_to_none=True)
        atom_contrib, bond_contrib, angle_contrib = model(data)
        all_sum = atom_contrib.sum() + bond_contrib.sum() + angle_contrib.sum()
        loss = loss_fn(all_sum, data.y[0][0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if parameters['run_ddp']:
            data.detach()
    return total_loss / len(loader)

def run_training(rank,data,parameters,model,ml=None):
    if parameters['run_ddp']:
        ddp_setup(rank,parameters['world_size'],parameters['ddp_backend'])
        if parameters['pre_training'] == False:
            if parameters['restart_training']:
                device = torch.device(ml.parameters['device'])
                ml.model.load_state_dict(torch.load(ml.parameters['restart_model_name'],map_location=device))
            model.to(parameters['device'])
            model = DDP(model, device_ids=[rank],find_unused_parameters=True)
        else: 
            model.to(ml.parameters['device'])
            device = torch.device(ml.parameters['device'])
            model.load_state_dict(torch.load(os.path.join(ml.parameters['pretrain_dir'], 'model_pre'),map_location=device))
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(data['training'][0], 'x_ang') else ['x_atm']
    if parameters['run_ddp']:
        loader_train = DataLoader(data['training'], batch_size=parameters['BATCH_SIZE'], shuffle=False,
                              follow_batch=follow_batch,sampler=DistributedSampler(data['training']))
        loader_valid = DataLoader(data['validation'], batch_size=parameters['BATCH_SIZE'], shuffle=False,
                                  sampler=DistributedSampler(data['validation']))
    else:
        loader_train = DataLoader(data['training'], batch_size=parameters['BATCH_SIZE'], shuffle=True,
                                  follow_batch=follow_batch)
        loader_valid = DataLoader(data['validation'], batch_size=parameters['BATCH_SIZE'], shuffle=False)

    L_train, L_valid = [], []
    min_loss_train = 1.0E30
    min_loss_valid = 1.0E30

    if rank == 0:
        stats_file = open(os.path.join(parameters['model_save_dir'],'loss.data'),'w')
        stats_file.write('Training_loss     Validation loss\n')
        stats_file.close()
    for ep in range(parameters['num_epochs']):
        if parameters['run_ddp']:
            loader_train.sampler.set_epoch(ep)
            loader_valid.sampler.set_epoch(ep)
        if rank == 0:
            print('Epoch ',ep,' of ',parameters['num_epochs'])
            sys.stdout.flush()
        loss_train = train(loader_train, model, parameters);
        L_train.append(loss_train)
        loss_valid = test_non_intepretable(loader_valid, model, parameters)
        L_valid.append(loss_valid)
        if rank == 0:
            stats_file = open(os.path.join(parameters['model_save_dir'], 'loss.data'), 'a')
            stats_file.write(str(loss_train) + '     ' + str(loss_valid) + '\n')
            stats_file.close()
        if loss_train < min_loss_train:
            min_loss_train = loss_train
            if loss_valid < min_loss_valid:
                min_loss_valid = loss_valid
                if parameters['remove_old_model']:
                    model_name = glob.glob(os.path.join(parameters['model_save_dir'], 'model_*'))
                    if len(model_name) > 0 and rank == 0:
                        os.remove(model_name[0])
                if rank == 0:
                    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    print('Min train loss: ', min_loss_train, ' min valid loss: ', min_loss_valid, ' time: ', now)
                    if parameters['run_ddp']:
                        torch.save(model.module.state_dict(), os.path.join(parameters['model_save_dir'], 'model_' + str(now)))
                    else:
                        torch.save(model.state_dict(), os.path.join(parameters['model_save_dir'], 'model_' + str(now)))
        if loss_train < parameters['train_tolerance'] and loss_valid < parameters['train_tolerance']:
            if rank == 0:
                print('Validation and training losses satisy set tolerance...exiting training loop...')
            break
    if parameters['run_ddp']:
        ddp_destroy()
def run_pre_training(rank,data,parameters,model):
    if parameters['run_ddp']:
        ddp_setup(rank, parameters['world_size'],parameters['ddp_backend'])
        model.to('cuda')
        #model.cuda(rank % torch.cuda.device_count())
        model = DDP(model, device_ids=[rank],find_unused_parameters=True)

    follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(data['training'][0], 'x_ang') else ['x_atm']
    if parameters['run_ddp']:
        loader_train = DataLoader(data['training'], batch_size=parameters['BATCH_SIZE'], shuffle=False,
                              follow_batch=follow_batch,sampler=DistributedSampler(data['training']))
    else:
        loader_train = DataLoader(data['training'], batch_size=parameters['BATCH_SIZE'], shuffle=True,
                                  follow_batch=follow_batch)
    L_train = []
    min_loss_train = 1.0E30

    if rank == 0:
        stats_file = open(os.path.join(parameters['pretrain_dir'], 'loss.data'), 'w')
        stats_file.write('Training_loss\n')
        stats_file.close()
    for ep in range(parameters['num_epochs']):
        if parameters['run_ddp']:
            loader_train.sampler.set_epoch(ep)
        if rank == 0:
            print('Epoch ', ep, ' of ', parameters['num_epochs'])
            sys.stdout.flush()
        loss_train = train(loader_train, model, parameters);
        L_train.append(loss_train)
        if rank == 0:
            stats_file = open(os.path.join(parameters['pretrain_dir'], 'loss.data'), 'a')
            stats_file.write(str(loss_train) + '\n')
            stats_file.close()
        if loss_train < min_loss_train:
            if rank == 0:
                print('Min train loss: ', min_loss_train)
            min_loss_train = loss_train
            model_name = glob.glob(os.path.join(parameters['pretrain_dir'], 'model_*'))
            if len(model_name) > 0 and rank == 0:
                os.remove(model_name[0])
            if rank == 0:
                if parameters['run_ddp']:
                    torch.save(model.module.state_dict(), os.path.join(parameters['pretrain_dir'], 'model_pre'))
                else:
                    torch.save(model.state_dict(), os.path.join(parameters['pretrain_dir'], 'model_pre'))
        if loss_train < parameters['train_tolerance']:
            if rank == 0:
                print('Pre-Training loss satisfies set tolerance...exiting training loop...')
            break
    if parameters['run_ddp']:
        ddp_destroy()







