from torch_geometric.loader import DataLoader
from torch import nn
import torch

from .testing import test_intepretable, test_non_intepretable

from tqdm.notebook import trange
from datetime import datetime
import glob
import sys
import os

def train(loader,model,parameters,PIN_MEMORY=False):
    model.train()
    total_loss = 0.0
    optimizer = torch.optim.AdamW(model.processor.parameters(), lr=parameters['LEARN_RATE'])
    model = model.to(parameters['device'])
    loss_fn = torch.nn.MSELoss()
    for data in loader:
        optimizer.zero_grad(set_to_none=True)
        data = data.to(parameters['device'], non_blocking=PIN_MEMORY)
        encoding = model.encoder(data)
        proc = model.processor(encoding)
        atom_contrib, bond_contrib, angle_contrib = model.decoder(proc)

        all_sum = atom_contrib.sum() + bond_contrib.sum() + angle_contrib.sum()

        loss = loss_fn(all_sum, data.y[0][0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def run_training(data,parameters,model):
    follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(data['training'][0], 'x_ang') else ['x_atm']
    loader_train = DataLoader(data['training'], batch_size=parameters['BATCH_SIZE'], shuffle=True, follow_batch=follow_batch)
    loader_valid = DataLoader(data['validation'], batch_size=parameters['BATCH_SIZE'], shuffle=False)

    print(f'Number of training data: {len(loader_train.dataset)}')
    print(f'Number of validation data: {len(loader_valid.dataset)}')

    L_train, L_valid = [], []
    min_loss_train = 1.0E30
    min_loss_valid = 1.0E30

    stats_file = open(os.path.join(parameters['model_dir'],'loss.data'),'w')
    stats_file.write('Training_loss     Validation loss\n')
    stats_file.close()
    for ep in range(parameters['num_epochs']):
        stats_file = open(os.path.join(parameters['model_dir'], 'loss.data'), 'a')
        print('Epoch ',ep,' of ',parameters['num_epochs'])
        sys.stdout.flush()
        loss_train = train(loader_train, model, parameters);
        L_train.append(loss_train)
        loss_valid = test_non_intepretable(loader_valid, model, parameters)
        L_valid.append(loss_valid)
        stats_file.write(str(loss_train) + '     ' + str(loss_valid) + '\n')
        if loss_train < min_loss_train:
            min_loss_train = loss_train
            if loss_valid < min_loss_valid:
                min_loss_valid = loss_valid
                if parameters['remove_old_model']:
                    model_name = glob.glob(os.path.join(parameters['model_dir'], 'model_*'))
                    if len(model_name) > 0:
                        os.remove(model_name[0])
                now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                print('Min train loss: ', min_loss_train, ' min valid loss: ', min_loss_valid, ' time: ', now)
                torch.save(model.state_dict(), os.path.join(parameters['model_dir'], 'model_' + str(now)))
        stats_file.close()
        if loss_train < parameters['train_tolerance'] and loss_valid < parameters['train_tolerance']:
            print('Validation and training losses satisy set tolerance...exiting training loop...')
            break
def run_pre_training(data,parameters,model):
    follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(data['training'][0], 'x_ang') else ['x_atm']
    loader_train = DataLoader(data['training'], batch_size=parameters['BATCH_SIZE'], shuffle=True,
                              follow_batch=follow_batch)
    print(f'Number of training data: {len(loader_train.dataset)}')
    L_train = []
    min_loss_train = 1.0E30

    stats_file = open(os.path.join(parameters['pretrain_dir'], 'loss.data'), 'w')
    stats_file.write('Training_loss\n')
    stats_file.close()
    for ep in range(parameters['num_epochs']):
        stats_file = open(os.path.join(parameters['pretrain_dir'], 'loss.data'), 'a')
        print('Epoch ', ep, ' of ', parameters['num_epochs'])
        sys.stdout.flush()
        loss_train = train(loader_train, model, parameters);
        L_train.append(loss_train)
        stats_file.write(str(loss_train) + '\n')
        if loss_train < min_loss_train:
            min_loss_train = loss_train
            model_name = glob.glob(os.path.join(parameters['pretrain_dir'], 'model_*'))
            if len(model_name) > 0:
                os.remove(model_name[0])
            print('Min train loss: ', min_loss_train)
            torch.save(model.state_dict(), os.path.join(parameters['pretrain_dir'], 'model_pre'))
        stats_file.close()
        if loss_train < parameters['train_tolerance']:
            print('Pre-Training loss satisfies set tolerance...exiting training loop...')
            break







