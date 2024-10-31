from ..utilities.rankings import organize_rankings_atomic, organize_rankings_generic
from .utils.distributed import reduce_tensor
from torch import nn
import torch

import numpy as np
import os

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

@torch.no_grad()
def test_non_intepretable(loader,model,parameters,ind_fn='all'):
    model.eval()
    loss_fn = parameters['model_dict']['loss_func']
    total_loss = 0.0
    loss_accum = parameters['model_dict']['accumulate_loss'][2]
    if parameters['io_dict']['write_indv_pred']:
        of = open(os.path.join(parameters['io_dict']['results_dir'],ind_fn + '_indv_pred.data'),'w')
        of.write('# True_y          Pred_y          \n')
    for data in loader:
        data = data.to(parameters['device_dict']['device'], non_blocking=parameters['device_dict']['pin_memory'])
        pred = model(data)
        preds, y = accumulate_predictions(pred,data,loss_accum)
        loss = loss_fn(preds,y)
        total_loss += loss.item()

        if parameters['io_dict']['write_indv_pred']:
            of.write(str(data.y.item()) + '          ' + str(preds.item()) + '\n')
    if parameters['io_dict']['write_indv_pred']:
        of.close()
    if parameters['device_dict']['run_ddp']:
        total_loss = reduce_tensor(torch.tensor(total_loss).to(parameters['device_dict']['device'])).item()
    return total_loss / (len(loader)*parameters['device_dict']['world_size'])

@torch.no_grad()
def predict_non_intepretable(loader,model,parameters,ind_fn='all'):
    model.eval()
    if parameters['io_dict']['write_indv_pred']:
        of = open(os.path.join(parameters['io_dict']['results_dir'], ind_fn + '_indv_pred.data'), 'w')
        of.write('#Pred_y          \n')
    all_preds = []
    for i,data in enumerate(loader):
        print('predicting on structure ', i)
        data = data.to(parameters['device_dict']['device'], non_blocking=parameters['device_dict']['pin_memory'])
        pred = model(data)
        for p in pred:
            if preds is None:
                preds = p.sum()
            else:
                preds += p.sum()
        all_preds.append(preds)
        if parameters['io_dict']['write_indv_pred']:
            of.write(str(preds.item()) + '\n')
    if parameters['io_dict']['write_indv_pred']:
        of.close()
    return all_preds

@torch.no_grad()
def predict_intepretable(loader,model,parameters,ind_fn='all'):
    model.eval()
    if parameters['io_dict']['write_indv_pred']:
        of = open(os.path.join(parameters['io_dict']['results_dir'], ind_fn + '_indv_pred.data'), 'w')
        of.write('#Pred_y          \n')
    all_preds = []
    for i,data in enumerate(loader):
        print('predicting on structure ',i)
        data = data.to(parameters['device_dict']['device'], non_blocking=parameters['device_dict']['pin_memory'])
        pred = model(data)
        preds = None
        for p in pred:
            if preds is None:
                preds = p.sum()
            else:
                preds += p.sum()
        all_preds.append(preds)
        if hasattr(data, 'x_atm_batch'):
            amounts = [data['atm_amounts'],data['bnd_amounts']]
            node_G_contrib = (pred[0].cpu() / preds.cpu()).flatten().numpy()
            node_A_contrib = (pred[1].cpu() / preds.cpu()).flatten().numpy()
            if hasattr(data, 'x_ang'):
                amounts.append(data['ang_amounts'])
                edge_A_contrib = (pred[2].cpu() / preds.cpu()).flatten().numpy()

                i_ng, x_na, i_na, x_ea, i_ea, x_ng = organize_rankings_atomic(data.cpu(), data.x_atm.cpu(), data.edge_index_G.cpu(),
                                                                      data.edge_index_A.cpu(),atom_mode='')
            else:
                i_ng, x_na, i_na, x_ea, i_ea, x_ng = organize_rankings_atomic(data.cpu(), data.x_atm.cpu(), data.edge_index_G.cpu(),
                                                                      None)
                amounts.append([])
        elif hasattr(data, 'node_G_batch'):
            amounts = [data['node_G_amounts'], data['node_A_amounts']]
            node_G_contrib = (pred[0].cpu() / preds.cpu()).flatten().numpy()
            node_A_contrib = (pred[1].cpu() / preds.cpu()).flatten().numpy()
            if hasattr(data, 'edge_A'):
                amounts.append(data['edge_A_amounts'])
                edge_A_contrib = (pred[2].cpu() / preds.cpu()).flatten().numpy()

                i_ng, x_na, i_na, x_ea, i_ea, x_ng = organize_rankings_generic(data.cpu(), data.node_G.cpu(),
                                                                                data.edge_index_G.cpu(),
                                                                                data.edge_index_A.cpu())
            else:
                i_ng, x_na, i_na, x_ea, i_ea, x_ng = organize_rankings_generic(data.cpu(), data.node_G.cpu(),
                                                                                data.edge_index_G.cpu())
                amounts.append([])

        ranked_data = dict(
                           ng=x_ng,
                           na=x_na,
                           ea=x_ea,
                           ng_data=[],
                           na_data=[],
                           ea_data=[],
                           i_ng_data=i_ng,
                           i_na_data=i_na,
                           i_ea_data=i_ea,
                           ng_amounts=amounts[0],
                           na_amounts=amounts[1],
                           ea_amounts=amounts[2]
                           )
        for typ in i_ng:
            for a in typ:
                ranked_data['ng_data'].append(node_G_contrib[a])
        for b in i_na:
            ranked_data['na_data'].append(node_A_contrib[b])
        if hasattr(data, 'x_ang') or hasattr(data, 'edge_A'):
            for a in i_ea:
                ranked_data['ea_data'].append(edge_A_contrib[a])
        np.save(os.path.join(parameters['io_dict']['results_dir'], str(i) + '_rankings.npy'), ranked_data)
        if parameters['io_dict']['write_indv_pred']:
            of.write(str(preds.item()) + '\n')
    if parameters['io_dict']['write_indv_pred']:
        of.close()
    return all_preds





