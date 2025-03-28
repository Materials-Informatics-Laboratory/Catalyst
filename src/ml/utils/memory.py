import torch
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None  # Model has no parameters

def clear_torch_memory():
    try:
        gc.collect()
        torch.cuda.empty_cache()
    except:
        print('Clearing cuda cache failed...')
        pass
def change_model_device(model,device):
    try:
        model.to(device)
    except:
        try:
            model.model.to(device)
        except:
            print('No model found...')
            exit(0)

