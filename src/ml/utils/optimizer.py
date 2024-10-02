import torch

def set_optimizer(parameters):
    optimzer = None
    if parameters['model_dict']['optimizer_params']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW([parameters['model_dict']['optimizer_params']['params_group']])
    elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Adadelta':
        optimizer = torch.optim.Adadelta([parameters['model_dict']['optimizer_params']['params_group']])
    elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Adagrad':
        optimizer = torch.optim.Adagrad([parameters['model_dict']['optimizer_params']['params_group']])
    elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam([parameters['model_dict']['optimizer_params']['params_group']])
    elif parameters['model_dict']['optimizer_params']['optimizer'] == 'SparseAdam':
        optimizer = torch.optim.SparseAdam([parameters['model_dict']['optimizer_params']['params_group']])
    elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Adamax':
        optimizer = torch.optim.Adamax([parameters['model_dict']['optimizer_params']['params_group']])
    elif parameters['model_dict']['optimizer_params']['optimizer'] == 'ASGD':
        optimizer = torch.optim.ASGD([parameters['model_dict']['optimizer_params']['params_group']])
    elif parameters['model_dict']['optimizer_params']['optimizer'] == 'LBFGS':
        optimizer = torch.optim.LBFGS([parameters['model_dict']['optimizer_params']['params_group']])
    elif parameters['model_dict']['optimizer_params']['optimizer'] == 'NAdam':
        optimizer = torch.optim.NAdam([parameters['model_dict']['optimizer_params']['params_group']])
    elif parameters['model_dict']['optimizer_params']['optimizer'] == 'RAdam':
        optimizer = torch.optim.RAdam([parameters['model_dict']['optimizer_params']['params_group']])
    elif parameters['model_dict']['optimizer_params']['optimizer'] == 'RMSprop':
        optimizer = torch.optim.RMSprop([parameters['model_dict']['optimizer_params']['params_group']])
    elif parameters['model_dict']['optimizer_params']['optimizer'] == 'Rprop':
        optimizer = torch.optim.Rprop([parameters['model_dict']['optimizer_params']['params_group']])
    elif parameters['model_dict']['optimizer_params']['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD([parameters['model_dict']['optimizer_params']['params_group']])

    return optimizer
