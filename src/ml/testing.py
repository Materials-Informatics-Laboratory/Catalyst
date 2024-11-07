from ..utilities.rankings import organize_rankings_atomic, organize_rankings_generic
from .utils.predict import accumulate_predictions
from .utils.distributed import reduce_tensor
from ..io.io import save_dictionary
from torch import nn
import torch

import numpy as np
import os

@torch.no_grad()
def test_non_intepretable(loader,model,parameters,ind_fn='all'):
    model.eval()
    loss_fn = parameters['model_dict']['loss_func']
    total_loss = 0.0
    loss_accum = parameters['model_dict']['accumulate_loss'][2]
    if parameters['io_dict']['write_indv_pred']:
        values = [[],[],[]]
    for data in loader:
        data = data.to(parameters['device_dict']['device'], non_blocking=parameters['device_dict']['pin_memory'])
        pred = model(data)
        preds, y = accumulate_predictions(pred,data,loss_accum)
        loss = loss_fn(preds,y).item()
        total_loss += loss

        if parameters['io_dict']['write_indv_pred']:
            values[0].append(preds.item())
            values[1].append(y.item())
            values[2].append(loss)
    if parameters['io_dict']['write_indv_pred']:
        test_info = {
            'pred':values[0],
            'y':values[1],
            'loss':values[2],
            'loss_fn':parameters['model_dict']['accumulate_loss'][2]
        }
        save_dictionary(fname=os.path.join(parameters['io_dict']['results_dir'],ind_fn + '_indv_pred.data'),
                        data=test_info)
    return total_loss / len(loader)

@torch.no_grad()
def predict_non_intepretable(loader,model,parameters,ind_fn='all'):
    model.eval()
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
        test_info = {
            'pred': all_preds,
        }
        save_dictionary(fname=os.path.join(parameters['io_dict']['results_dir'], ind_fn + '_indv_pred.data'),
                        data=test_info)
    return all_preds

@torch.no_grad()
def predict_intepretable(loader,model,parameters,ind_fn='all'):
    model.eval()
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
        save_dictionary(fname=os.path.join(parameters['io_dict']['results_dir'], str(i) + '_rankings.data'),
                        data=ranked_data)
    if parameters['io_dict']['write_indv_pred']:
        test_info = {
            'pred': all_preds,
        }
        save_dictionary(fname=os.path.join(parameters['io_dict']['results_dir'], ind_fn + '_indv_pred.data'),
                        data=test_info)
    return all_preds





