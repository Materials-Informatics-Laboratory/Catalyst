from ..utilities.rankings import organize_rankings_atomic, organize_rankings_generic
from ml.gnn.modules.utils.predict import accumulate_predictions
from .utils.loss import loss_setup
from .utils.distributed import reduce_tensor, combine_dicts_across_gpus, ddp_destroy, ddp_setup

from ..data.utils import load_dictionary, save_dictionary
import torch

import glob as glob
import os


@torch.no_grad()
def test_non_intepretable_external(cat,ind_fn='all',rank=0):
    parameters = cat.parameters
    if parameters['device_dict']['run_ddp']:
        ddp_setup(rank, parameters['device_dict']['world_size'], parameters['device_dict']['ddp_backend'])

    if rank == 0:
        print('Reading data...')

    if cat.parameters['io_dict']['graph_read_format'] != -1:
        files = glob.glob(os.path.join(parameters['io_dict']['data_dir'], '*'))
        graphs = [None]*len(files)
        for i in range(len(files)):
            graphs[i] = torch.load(files[i])
    else:
        graphs = load_dictionary(glob.glob(os.path.join(cat.parameters['io_dict']['data_dir'], 'graphs.data'))[0])['graphs']

    data = dict(validation = graphs)
    model, model_data = setup_model(cat, rank=rank,load=True)
    loader_valid = setup_dataloader(data=data,cat=cat,mode=2)
    if rank == 0:
        print('Testing...')
   # loss = parameters['model_dict']['model']
    if parameters['device_dict']['run_ddp']:
        ddp_destroy()
    return loss




@torch.no_grad()
def predict_external(cat,ind_fn='all',rank=0,interpretable=0):
    parameters = cat.parameters
    if parameters['device_dict']['run_ddp']:
        ddp_setup(rank, parameters['device_dict']['world_size'], parameters['device_dict']['ddp_backend'])

    if rank == 0:
        print('Reading data...')

    if cat.parameters['io_dict']['graph_read_format'] != 2:
        files = glob.glob(os.path.join(parameters['io_dict']['data_dir'], '*'))
        graphs = [None]*len(files)
        for i in range(len(files)):
            graphs[i] = torch.load(files[i])
    else:
        graphs = load_dictionary(glob.glob(os.path.join(cat.parameters['io_dict']['data_dir'], 'graphs.data'))[0])['graphs']

    data = dict(validation = graphs)
    model = setup_model(cat, rank=rank,load=True)
    loader_valid = setup_dataloader(data=data,cat=cat,mode=2)
    if rank == 0:
        print('Predicting...')
    if interpretable:
        predict_interpretable(loader=loader_valid,
                                          model=model,
                                          parameters=parameters,
                                          ind_fn=ind_fn, rank=rank)
    else:
        predict_non_interpretable(loader=loader_valid,
                                          model=model,
                                          parameters=parameters,
                                          ind_fn=ind_fn,rank=rank)
    if parameters['device_dict']['run_ddp']:
        ddp_destroy()

@torch.no_grad()
def predict_non_interpretable(loader,model,parameters,ind_fn='all',rank=0):
    model.eval()
    all_preds = []
    for i,data in enumerate(loader):
        if rank == 0:
            print('predicting on structure (rank = 0) ', i)
        data = data.to(parameters['device_dict']['device'], non_blocking=parameters['device_dict']['pin_memory'])
        pred = model(data)
        preds, y, vec = accumulate_predictions(pred, data, 'exact')
        all_preds.append(preds)
    if parameters['io_dict']['write_indv_pred']:
        test_info = {
            'pred': all_preds,
        }
        if parameters['device_dict']['run_ddp']:
            test_info = combine_dicts_across_gpus(test_info)
        if rank == 0:
            save_dictionary(fname=os.path.join(parameters['io_dict']['results_dir'], ind_fn + '_indv_pred.data'),
                        data=test_info)

@torch.no_grad()
def predict_interpretable(loader,model,parameters,ind_fn='all',rank=0):
    model.eval()
    all_preds = []
    for i,data in enumerate(loader):
        if rank == 0:
            print('predicting on structure ',i,' of ',len(loader))
        data = data.to(parameters['device_dict']['device'], non_blocking=parameters['device_dict']['pin_memory'])
        pred = model(data)
        preds, y, vec = accumulate_predictions(pred, data, 'exact')
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
        if parameters['device_dict']['run_ddp']:
            ranked_data = combine_dicts_across_gpus(ranked_data)
        if rank == 0:
            save_dictionary(fname=os.path.join(parameters['io_dict']['results_dir'], str(i) + '_rankings.data'),
                        data=ranked_data)
    if parameters['io_dict']['write_indv_pred']:
        test_info = {
            'pred': all_preds,
        }
        if parameters['device_dict']['run_ddp']:
            test_info = combine_dicts_across_gpus(test_info)
        if rank == 0:
            save_dictionary(fname=os.path.join(parameters['io_dict']['results_dir'], ind_fn + '_indv_pred.data'),
                        data=test_info)





