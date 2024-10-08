from ..utilities.rankings import organize_rankings
from .utils.distributed import reduce_tensor
from torch import nn
import torch

import numpy as np
import os

@torch.no_grad()
def test_intepretable(loader,model,parameters):
    model.eval()
    loss_fn = parameters['model_dict']['loss_func']
    total_loss = 0.0
    for i,data in enumerate(loader):
        print('Testing on data point ',i)
        data = data.to(parameters['device_dict']['device'], non_blocking=parameters['device_dict']['pin_memory'])
        pred = model(data)

        atom_contrib = (pred[0].cpu() / all_sum.cpu()).flatten().numpy()
        bond_contrib = (pred[1].cpu() / all_sum.cpu()).flatten().numpy()
        if hasattr(data,'x_ang'):
            angle_contrib = (pred[2].cpu() / all_sum.cpu()).flatten().numpy()

            i_atm, x_bnd, i_bnd, x_ang, i_ang = organize_rankings(data.cpu(),data.x_atm.cpu(), data.edge_index_G.cpu(),
            data.edge_index_A.cpu(), parameters['elements'])
        else:
            i_atm, x_bnd, i_bnd, x_ang, i_ang = organize_rankings(data.cpu(), data.x_atm.cpu(), data.edge_index_G.cpu(),
                                                                  None, parameters['elements'])

        ranked_data = dict(y=[data.y.item(), pred.item()],
            atms = parameters['elements'],
            bnds = x_bnd,
            angs = x_ang,
            atm_data = [],
            bnd_data = [],
            ang_data = [],
            i_atm_data = i_atm,
            i_bnd_data = i_bnd,
            i_ang_data = i_ang,
            atm_amounts = data['atm_amounts'],
            bnd_amounts = data['bnd_amounts'],
            ang_amounts = data['ang_amounts']
        )
        for typ in i_atm:
            for a in typ:
                ranked_data['atm_data'].append(atom_contrib[a])
        for b in i_bnd:
            ranked_data['bnd_data'].append(bond_contrib[b])
        if hasattr(data,'x_ang'):
            for a in i_ang:
                ranked_data['ang_data'].append(angle_contrib[a])
        np.save(os.path.join(parameters['io_dict']['results_dir'], str(i) + '_rankings.npy'), ranked_data)

        loss = loss_fn(all_sum, data.y)
        total_loss += loss

    if parameters['device_dict']['run_ddp']:
        total_loss = reduce_tensor(torch.tensor(total_loss).to(parameters['device_dict']['device'])).item()
    return total_loss

@torch.no_grad()
def test_non_intepretable(loader,model,parameters,ind_fn='all'):
    model.eval()
    loss_fn = parameters['model_dict']['loss_func']
    total_loss = 0.0
    if parameters['io_dict']['write_indv_pred']:
        of = open(os.path.join(parameters['io_dict']['results_dir'],ind_fn + '_indv_pred.data'),'w')
        of.write('# True_y          Pred_y          \n')
    for data in loader:
        data = data.to(parameters['device_dict']['device'], non_blocking=parameters['device_dict']['pin_memory'])
        pred = model(data)

        if parameters['model_dict']['accumulate_loss'][2] == 'exact':
            if hasattr(data, 'x_atm_batch'):
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

                y = data.y.flatten()
            else:
                '''
                Implement generic routines here for non alignn systems
                '''
                pass
        elif parameters['model_dict']['accumulate_loss'][2] == 'sum':
            if hasattr(data, 'x_ang_batch') and hasattr(data, 'x_ang_batch'):
                preds = pred[0].sum() + pred[1].sum() + pred[2].sum()
            else:
                preds = pred[0].sum() + pred[1].sum()
            y = data.y.flatten().sum()

        loss = loss_fn(preds,y)
        total_loss += loss.item()

        if parameters['io_dict']['write_indv_pred']:
            of.write(str(data.y.item()) + '          ' + str(preds.item()) + '\n')
    if parameters['io_dict']['write_indv_pred']:
        of.close()
    if parameters['device_dict']['run_ddp']:
        total_loss = reduce_tensor(torch.tensor(total_loss).to(parameters['device_dict']['device'])).item()
    return total_loss

@torch.no_grad()
def predict_non_intepretable(loader,model,parameters,ind_fn='all'):
    model.eval()
    if parameters['io_dict']['write_indv_pred']:
        of = open(os.path.join(parameters['io_dict']['results_dir'], ind_fn + '_indv_pred.data'), 'w')
        of.write('#Pred_y          \n')
    preds = []
    for i,data in enumerate(loader):
        print('predicting on structure ', i)
        data = data.to(parameters['device_dict']['device'], non_blocking=parameters['device_dict']['pin_memory'])
        pred = model(data)
        all_sum = 0.0
        for p in pred:
            all_sum += p.sum()
        preds.append(all_sum)
        if parameters['io_dict']['write_indv_pred']:
            of.write(str(all_sum.item()) + '\n')
    if parameters['io_dict']['write_indv_pred']:
        of.close()
    return preds

@torch.no_grad()
def predict_intepretable(loader,model,parameters,ind_fn='all'):
    model.eval()
    if parameters['io_dict']['write_indv_pred']:
        of = open(os.path.join(parameters['io_dict']['results_dir'], ind_fn + '_indv_pred.data'), 'w')
        of.write('#Pred_y          \n')
    preds = []
    for i,data in enumerate(loader):
        print('predicting on structure ',i)
        data = data.to(parameters['device_dict']['device'], non_blocking=parameters['device_dict']['pin_memory'])
        pred = model(data)
        all_sum = 0.0
        for p in pred:
            all_sum += p.sum()

        atom_contrib = (pred[0].cpu() / all_sum.cpu()).flatten().numpy()
        bond_contrib = (pred[1].cpu() / all_sum.cpu()).flatten().numpy()
        if hasattr(data, 'x_ang'):
            angle_contrib = (pred[2].cpu() / all_sum.cpu()).flatten().numpy()

            i_atm, x_bnd, i_bnd, x_ang, i_ang, elements = organize_rankings(data.cpu(), data.x_atm.cpu(), data.edge_index_G.cpu(),
                                                                  data.edge_index_A.cpu(),atom_mode='')
        else:
            i_atm, x_bnd, i_bnd, x_ang, i_ang, elements = organize_rankings(data.cpu(), data.x_atm.cpu(), data.edge_index_G.cpu(),
                                                                  None)

        ranked_data = dict(
                           atms=elements,
                           bnds=x_bnd,
                           angs=x_ang,
                           atm_data=[],
                           bnd_data=[],
                           ang_data=[],
                           i_atm_data=i_atm,
                           i_bnd_data=i_bnd,
                           i_ang_data=i_ang,
                           atm_amounts=data['atm_amounts'],
                           bnd_amounts=data['bnd_amounts'],
                           ang_amounts=data['ang_amounts']
                           )
        for typ in i_atm:
            for a in typ:
                ranked_data['atm_data'].append(atom_contrib[a])
        for b in i_bnd:
            ranked_data['bnd_data'].append(bond_contrib[b])
        if hasattr(data, 'x_ang'):
            for a in i_ang:
                ranked_data['ang_data'].append(angle_contrib[a])
        np.save(os.path.join(parameters['io_dict']['results_dir'], str(i) + '_rankings.npy'), ranked_data)

        preds.append(all_sum)
        if parameters['io_dict']['write_indv_pred']:
            of.write(str(all_sum.item()) + '\n')
    if parameters['io_dict']['write_indv_pred']:
        of.close()
    return preds





