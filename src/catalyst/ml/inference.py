from ..utilities.rankings import organize_rankings_atomic, organize_rankings_generic
from .utils.loss import loss_setup
from .utils.distributed import ddp_destroy, ddp_setup, ddp_model, validate_ddp_configuration

from ..data.utils import load_dictionary, save_dictionary
import torch

import glob as glob
import os

def setup_inference(rank,model_name,cat=None):
    if rank == 0:
        print('Running inference...')

    parameters = cat.parameters
    if hasattr(cat, "validate_parameters"):
        model = parameters["model_dict"].get("model")
        if getattr(cat, "task", None) is None:
            model_task = getattr(model, "_catalyst_task", None) if model is not None else None
            if model_task is None and model is not None and hasattr(model, "model"):
                model_task = getattr(model.model, "_catalyst_task", None)
            if model_task is not None:
                cat.set_task(model_task)
            else:
                # Backward-compatible reconstruction for older/direct builder workflows.
                from .gnn.tasks import task_from_parameters
                cat.set_task(task_from_parameters(parameters))
            parameters = cat.parameters
        cat.validate_parameters(
            stage="model",
            model=parameters["model_dict"].get("model"),
            rank=rank,
        )
        parameters = cat.parameters
    validate_ddp_configuration(parameters, rank=rank)
    if parameters['device_dict']['run_ddp']:
        ddp_setup(rank, parameters['device_dict']['world_size'], parameters['device_dict']['ddp_backend'])

    parameters['model_dict']['model'].load_checkpoint(fname=model_name)
    parameters['model_dict']['model'].load_data(parameters,
           format=parameters['io_dict']['graph_read_format'], rank=rank,load_training=False,
                                                samples_file=os.path.join(parameters['io_dict']['samples_dir'],'test_data.npy'))

@torch.no_grad()
def run_inference(model_name,rank=0,cat=None,test=False):
    parameters = cat.parameters
    setup_inference(rank=rank,model_name=model_name, cat=cat)

    model = parameters['model_dict']['model']
    model.device = parameters['device_dict']['device']
    model.model.to(model.device)
    if hasattr(model, "configure_numeric_performance"):
        model.configure_numeric_performance(parameters)
    if parameters['device_dict']['run_ddp']:
        model.model = ddp_model(
            model=model.model,
            find_unused_parameters=parameters['device_dict']['find_unused_parameters'],
            rank=rank,
            batchnorm=parameters['model_dict'].get('batchnorm', False),
            gradient_as_bucket_view=parameters['device_dict'].get('ddp_gradient_as_bucket_view', False),
            static_graph=parameters['device_dict'].get('ddp_static_graph', False),
            bucket_cap_mb=parameters['device_dict'].get('ddp_bucket_cap_mb', None),
        )
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
