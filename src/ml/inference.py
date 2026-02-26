from ..utilities.rankings import organize_rankings_atomic, organize_rankings_generic
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

