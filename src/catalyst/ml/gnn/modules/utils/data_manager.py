from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from .....graph.graph import Atomic_Graph_Data, Generic_Graph_Data, Graph_Data
import math
import warnings


def _shared_sampler_seed(parameters):
    """Return one deterministic seed shared by every DDP rank."""
    sampling = parameters.get("sampling_dict", {}) or {}
    return int(sampling.get("sampling_seed", 0))


def _follow_batch_fields(example):
    if isinstance(example, Atomic_Graph_Data):
        return ["x_atm", "x_bnd", "x_ang"] if getattr(example, "x_ang", None) is not None else ["x_atm", "x_bnd"]
    if isinstance(example, Generic_Graph_Data):
        return ["node_G", "node_A", "edge_A"] if getattr(example, "edge_A", None) is not None else ["node_G", "node_A"]
    if isinstance(example, Graph_Data):
        return []
    return []


def _worker_kwargs(parameters):
    cfg = parameters.get("loader_dict", {}) or {}
    num_workers = int(cfg.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("loader_dict['num_workers'] must be >= 0.")
    kwargs = {"num_workers": num_workers}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", False))
        prefetch_factor = cfg.get("prefetch_factor", 2)
        if prefetch_factor is not None:
            prefetch_factor = int(prefetch_factor)
            if prefetch_factor < 1:
                raise ValueError("loader_dict['prefetch_factor'] must be >= 1.")
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def _maybe_prefetch(loader, parameters):
    cfg = parameters.get("loader_dict", {}) or {}
    if not bool(cfg.get("prefetch_to_device", False)):
        return loader
    if bool(parameters.get("device_dict", {}).get("run_ddp", False)):
        raise ValueError(
            "loader_dict['prefetch_to_device'] is currently supported for single-device "
            "training only. DDP already overlaps rank-local input work and requires direct "
            "access to its DistributedSampler."
        )

    device = str(parameters.get("device_dict", {}).get("device", "cpu"))
    if not device.startswith("cuda"):
        warnings.warn(
            "prefetch_to_device=True has no GPU benefit on a non-CUDA device; "
            "returning the ordinary DataLoader.",
            RuntimeWarning,
        )
        return loader

    try:
        from torch_geometric.loader import PrefetchLoader
    except ImportError as exc:
        raise ImportError(
            "prefetch_to_device=True requires a PyTorch Geometric version that "
            "provides torch_geometric.loader.PrefetchLoader."
        ) from exc
    return PrefetchLoader(loader, device=device)


def _dynamic_batch_loader(data, parameters, loader_params, follow_batch, common_kwargs):
    cfg = parameters.get("loader_dict", {}) or {}
    if bool(parameters.get("device_dict", {}).get("run_ddp", False)):
        raise ValueError(
            "Dynamic node/edge-budget batching is currently supported for single-device "
            "training only. Use batch_mode='graphs' with DDP."
        )

    mode = str(cfg.get("batch_mode", "graphs")).lower()
    if mode not in {"nodes", "edges"}:
        raise ValueError("loader_dict['batch_mode'] must be 'graphs', 'nodes', or 'edges'.")

    max_num = cfg.get("max_nodes" if mode == "nodes" else "max_edges")
    if max_num is None:
        raise ValueError(
            f"batch_mode={mode!r} requires loader_dict["
            f"{'max_nodes' if mode == 'nodes' else 'max_edges'}] to be set."
        )
    max_num = int(max_num)
    if max_num < 1:
        raise ValueError("Dynamic batch node/edge budget must be >= 1.")

    try:
        from torch_geometric.loader import DynamicBatchSampler
    except ImportError as exc:
        raise ImportError(
            "Dynamic node/edge batching requires a PyTorch Geometric version that "
            "provides torch_geometric.loader.DynamicBatchSampler."
        ) from exc

    sampler = DynamicBatchSampler(
        data,
        max_num=max_num,
        mode="node" if mode == "nodes" else "edge",
        shuffle=bool(loader_params.get("shuffle", False)),
        skip_too_big=bool(cfg.get("dynamic_batch_skip_too_big", False)),
        num_steps=cfg.get("dynamic_batch_num_steps", None),
    )
    loader = DataLoader(
        data,
        batch_sampler=sampler,
        follow_batch=follow_batch,
        **common_kwargs,
    )
    return _maybe_prefetch(loader, parameters)


def setup_dataloader(data, cat, loader_params):
    parameters = cat.parameters
    if data is None or len(data) == 0:
        raise ValueError("Cannot create a DataLoader from an empty dataset.")

    follow_batch = _follow_batch_fields(data[0])
    pin_memory = bool(parameters.get("device_dict", {}).get("pin_memory", False))
    common_kwargs = {"pin_memory": pin_memory, **_worker_kwargs(parameters)}

    batch_mode = str(parameters.get("loader_dict", {}).get("batch_mode", "graphs")).lower()
    if batch_mode in {"nodes", "edges"}:
        return _dynamic_batch_loader(
            data, parameters, loader_params, follow_batch, common_kwargs
        )
    if batch_mode != "graphs":
        raise ValueError("loader_dict['batch_mode'] must be 'graphs', 'nodes', or 'edges'.")

    if parameters["device_dict"]["run_ddp"]:
        sampler = DistributedSampler(
            data,
            shuffle=bool(loader_params["shuffle"]),
            seed=_shared_sampler_seed(parameters),
        )
        sampler.set_epoch(int(loader_params.get("epoch", 0)))
        loader = DataLoader(
            data,
            batch_size=math.ceil(
                loader_params["batch_size"] / parameters["device_dict"]["world_size"]
            ),
            follow_batch=follow_batch,
            sampler=sampler,
            **common_kwargs,
        )
        return loader

    loader = DataLoader(
        data,
        batch_size=loader_params["batch_size"],
        shuffle=loader_params["shuffle"],
        follow_batch=follow_batch,
        **common_kwargs,
    )
    return _maybe_prefetch(loader, parameters)
