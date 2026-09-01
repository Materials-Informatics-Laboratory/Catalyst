import gc

import torch


def optimizer_to(optim, device):
    """Move optimizer state tensors to ``device`` in-place."""
    for param in optim.state.values():
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
        return None


def clear_torch_memory():
    """Release Python garbage and any unused CUDA caching allocator blocks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def change_model_device(model, device):
    """Move a model/wrapper to ``device`` or raise a useful error.

    Older Catalyst code called ``exit(0)`` when both move attempts failed, which
    made a genuine model/device error look like a successful process exit.
    """
    try:
        model.to(device)
        return model
    except (AttributeError, TypeError, RuntimeError) as direct_error:
        wrapped = getattr(model, "model", None)
        if wrapped is None:
            raise RuntimeError(
                f"Failed to move model of type {type(model).__name__} to {device!r}."
            ) from direct_error
        try:
            wrapped.to(device)
            return model
        except (AttributeError, TypeError, RuntimeError) as wrapped_error:
            raise RuntimeError(
                f"Failed to move model or wrapped model to {device!r}."
            ) from wrapped_error
