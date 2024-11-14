from ..utilities.rankings import organize_rankings_atomic, organize_rankings_generic
from .utils.predict import accumulate_predictions
from .utils.distributed import reduce_tensor, combine_dicts_across_gpus, ddp_destroy, ddp_setup
from ..io.io import read_training_data, setup_model, setup_dataloader, save_model, save_dictionary
from torch import nn
import torch

import numpy as np
import glob as glob
import os

@torch.no_grad()
def test_non_intepretable_external(ml,ind_fn='all',rank=0):
    parameters = ml.parameters
    if parameters['device_dict']['run_ddp']:
        ddp_setup(rank, parameters['device_dict']['world_size'], parameters['device_dict']['ddp_backend'])

    if rank == 0:
        print('Reading data...')
    data = dict(validation = [torch.load(file_name) for file_name in
                  glob.glob(os.path.join(parameters['io_dict']['data_dir'], '*'))]
    )
    model = setup_model(ml, rank=rank,load=True)
    loader_valid = setup_dataloader(data=data,ml=ml,mode=2)
    if rank == 0:
        print('Testing...')
    loss = test_non_intepretable_internal(loader=loader_valid,
                                          model=model,
                                          parameters=parameters,
                                          ind_fn='all',rank=rank)
@torch.no_grad()
def test_non_intepretable_internal(loader,model,parameters,ind_fn='all',rank=0):
    model.eval()
    loss_fn = parameters['model_dict']['loss_func']
    epoch_loss = 0.0
    loss_accum = parameters['model_dict']['accumulate_loss'][2]
    if parameters['io_dict']['write_indv_pred']:
        values = [[],[],[]]
        gids = []
    for data in loader:
        data = data.to(parameters['device_dict']['device'], non_blocking=parameters['device_dict']['pin_memory'])
        pred = model(data)
        preds, y, vec = accumulate_predictions(pred, data, loss_accum)
        preds = preds.to(y.device)
        if vec:
            loss_list = [0.0] * len(preds)
            for i in range(len(preds)):
                loss_list[i] = loss_fn(preds[i], y[i])
        else:
            loss_list = [loss_fn(preds, y)]
        batch_loss = torch.sum(torch.stack(loss_list))
        epoch_loss += batch_loss.item()

        if parameters['io_dict']['write_indv_pred']:
            values[0].append(preds.tolist())
            values[1].append(y.tolist())
            values[2].append(loss_list)
            gids.append(data.gid)

    if parameters['io_dict']['write_indv_pred']:
        test_info = {
            'gids': gids,
            'pred': values[0],
            'y': values[1],
            'loss': values[2],
            'loss_fn': parameters['model_dict']['accumulate_loss'][2],
            'vec':vec
        }
        if parameters['device_dict']['run_ddp']:
            test_info = combine_dicts_across_gpus(test_info)
        if rank == 0:
            save_dictionary(fname=os.path.join(parameters['io_dict']['results_dir'],ind_fn + '_indv_pred.data'),
                            data=test_info)
    if parameters['device_dict']['run_ddp']:
        epoch_loss = reduce_tensor(torch.tensor(epoch_loss).to(parameters['device_dict']['device'])).item()
    return epoch_loss / (len(loader) * parameters['device_dict']['world_size'])

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





