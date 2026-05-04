from ..utilities.rankings import organize_rankings_atomic, organize_rankings_generic
from .utils.loss import loss_setup
from .utils.distributed import reduce_tensor, combine_dicts_across_gpus, ddp_destroy, ddp_setup, ddp_model

from ..data.utils import load_dictionary, save_dictionary
import torch

import glob as glob
import os

def setup_inference(rank,model_name,cat=None):
    if rank == 0:
        print('Running inference...')

    parameters = cat.parameters
    if parameters['device_dict']['run_ddp']:
        ddp_setup(rank, parameters['device_dict']['world_size'], parameters['device_dict']['ddp_backend'])

    parameters['model_dict']['model'].load_checkpoint(fname=model_name)
    parameters['model_dict']['model'].load_data(parameters,
           format=parameters['io_dict']['graph_read_format'], rank=rank,load_training=False,
                                                samples_file=os.path.join(parameters['io_dict']['samples_dir'],'test_data.npy'))

@torch.no_grad()
def run_inference(rank,model_name,cat=None,test=False):
    parameters = cat.parameters
    setup_inference(rank=rank,model_name=model_name, cat=cat)

    model = parameters['model_dict']['model']
    model.device = parameters['device_dict']['device']
    if parameters['device_dict']['run_ddp']:
        model.model = ddp_model(model=model.model,
                                find_unused_parameters=parameters['device_dict']['find_unused_parameters'],
                                rank=rank, batchnorm=parameters['model_dict']['batchnorm'])
    model.set_dataloader(cat=cat,training=False)
    if test:
        test_dict = model.validate(parameters=parameters, rank=rank)
    else:
        test_dict = model.predict(parameters=parameters, rank=rank)

    if test:
        pass
    else:
        if parameters['io_dict']['write_indv_pred']:
            if rank == 0:
                save_dictionary(fname=os.path.join(parameters['io_dict']['results_dir'], 'indv_pred.data'),
                                data=test_dict)
    return test_dict
