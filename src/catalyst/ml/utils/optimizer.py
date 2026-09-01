import inspect

import torch


_OPTIMIZERS = {
    "AdamW": torch.optim.AdamW,
    "Adadelta": torch.optim.Adadelta,
    "Adagrad": torch.optim.Adagrad,
    "Adam": torch.optim.Adam,
    "SparseAdam": torch.optim.SparseAdam,
    "Adamax": torch.optim.Adamax,
    "ASGD": torch.optim.ASGD,
    "LBFGS": torch.optim.LBFGS,
    "NAdam": torch.optim.NAdam,
    "RAdam": torch.optim.RAdam,
    "RMSprop": torch.optim.RMSprop,
    "Rprop": torch.optim.Rprop,
    "SGD": torch.optim.SGD,
}


def _optimizer_impl_kwargs(optimizer_cls, implementation, params):
    implementation = str(implementation or "default").lower()
    if implementation not in {"default", "auto", "fused", "foreach", "for_loop"}:
        raise ValueError(
            "optimizer_params['implementation'] must be one of "
            "'default', 'auto', 'fused', 'foreach', or 'for_loop'."
        )

    signature = inspect.signature(optimizer_cls.__init__)
    supports_fused = "fused" in signature.parameters
    supports_foreach = "foreach" in signature.parameters

    if implementation == "default":
        return {}
    if implementation == "for_loop":
        return {"foreach": False, "fused": False} if supports_fused and supports_foreach else ({"foreach": False} if supports_foreach else {})
    if implementation == "foreach":
        if not supports_foreach:
            raise ValueError(f"{optimizer_cls.__name__} does not expose a foreach implementation.")
        return {"foreach": True, **({"fused": False} if supports_fused else {})}
    if implementation == "fused":
        if not supports_fused:
            raise ValueError(f"{optimizer_cls.__name__} does not expose a fused implementation.")
        return {"fused": True, **({"foreach": False} if supports_foreach else {})}

    # auto: prefer fused on CUDA when supported, then foreach, otherwise let
    # PyTorch choose its ordinary implementation.
    first_param = next(iter(params), None)
    on_cuda = bool(first_param is not None and getattr(first_param, "is_cuda", False))
    if on_cuda and supports_fused:
        return {"fused": True, **({"foreach": False} if supports_foreach else {})}
    if supports_foreach:
        return {"foreach": True, **({"fused": False} if supports_fused else {})}
    return {}


def set_optimizer(parameters):
    """Create the configured PyTorch optimizer.

    ``params_group`` remains backward compatible with the historical Catalyst
    configuration.  Performance implementation controls live one level above it
    because ``fused``/``foreach`` are optimizer-constructor options rather than
    trainable parameter-group hyperparameters.
    """
    optimizer_cfg = parameters["model_dict"]["optimizer_params"]
    optimizer_name = optimizer_cfg.get("optimizer")
    optimizer_cls = _OPTIMIZERS.get(optimizer_name)
    if optimizer_cls is None:
        raise ValueError(
            f"Unsupported optimizer {optimizer_name!r}. Available: {sorted(_OPTIMIZERS)}"
        )

    group = dict(optimizer_cfg.get("params_group", {}) or {})
    params = group.pop("params", None)
    if params is None:
        raise ValueError("optimizer params_group must contain a 'params' iterable.")
    params = list(params)
    if not params:
        raise ValueError("Cannot construct an optimizer for an empty parameter list.")

    implementation = optimizer_cfg.get("implementation", "default")
    impl_kwargs = _optimizer_impl_kwargs(optimizer_cls, implementation, params)
    parameter_group = {"params": params, **group}
    return optimizer_cls([parameter_group], **impl_kwargs)
