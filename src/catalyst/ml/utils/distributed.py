from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import gc
import os


def set_spawn_method(parameters):
    """Use the spawn multiprocessing context for NCCL-backed DDP runs."""
    device_dict = parameters.get("device_dict", parameters)
    backend = str(device_dict.get("ddp_backend", "")).lower()
    if backend == "nccl":
        mp.set_start_method("spawn", force=True)


def validate_ddp_configuration(parameters, rank=None):
    """Validate Catalyst's supported CUDA DDP configuration early.

    Catalyst 2.2 advertises multi-GPU CUDA DDP.  Failing here gives a clear
    message instead of allowing a later NCCL/device error deep inside training.
    """
    device_dict = parameters.get("device_dict", parameters)
    if not bool(device_dict.get("run_ddp", False)):
        return

    backend = str(device_dict.get("ddp_backend", "nccl") or "nccl").lower()
    world_size = int(device_dict.get("world_size", 1))
    device = str(device_dict.get("device", "cuda"))

    if world_size < 1:
        raise ValueError("device_dict['world_size'] must be >= 1 for DDP.")
    if not device.startswith("cuda"):
        raise ValueError(
            "Catalyst 2.2 DDP currently supports CUDA devices only. "
            "Set device_dict['device']='cuda' (or cuda:<id>) or disable DDP."
        )
    if backend != "nccl" and backend != "gloo":
        raise ValueError(
            "Catalyst 2.2 CUDA DDP is validated for the NCCL or GLOO backends only; "
            f"received ddp_backend={backend!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("DDP was requested but torch.cuda.is_available() is False.")

    device_count = torch.cuda.device_count()
    if world_size > device_count:
        raise ValueError(
            f"DDP world_size={world_size} exceeds the {device_count} visible CUDA device(s)."
        )
    if rank is not None and not (0 <= int(rank) < world_size):
        raise ValueError(f"rank={rank} must satisfy 0 <= rank < world_size={world_size}.")


def ddp_setup(rank: int, world_size, backend):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA DDP setup requested but CUDA is unavailable.")
    if str(backend).lower() != "nccl" and str(backend).lower() != "gloo":
        raise ValueError("Catalyst CUDA DDP currently requires backend='nccl' or 'gloo'.")

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    torch.cuda.set_device(rank)
    init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method="env://?use_libuv=False",
    )


def ddp_model(
    model,
    find_unused_parameters,
    rank,
    batchnorm,
    *,
    gradient_as_bucket_view=False,
    static_graph=False,
    bucket_cap_mb=None,
):
    """Move a model to one CUDA rank and wrap it in DDP."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    model.to(device)

    if batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    kwargs = dict(
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=bool(find_unused_parameters),
        gradient_as_bucket_view=bool(gradient_as_bucket_view),
        static_graph=bool(static_graph),
    )
    if bucket_cap_mb is not None:
        kwargs["bucket_cap_mb"] = float(bucket_cap_mb)

    return DDP(model, **kwargs)


def cuda_destroy():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ddp_destroy():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        finally:
            destroy_process_group()


def reduce_tensor(tensor):
    """Average a tensor across the active distributed process group."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor.clone().detach()
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def _merge_gathered_values(values):
    """Merge rank-local result values into one stable public representation."""
    if all(isinstance(value, list) for value in values):
        merged = []
        for value in values:
            merged.extend(value)
        return merged
    if all(isinstance(value, tuple) for value in values):
        merged = []
        for value in values:
            merged.extend(value)
        return tuple(merged)
    if all(isinstance(value, dict) for value in values):
        return _merge_result_dicts(values)
    scalar_types = (str, int, float, bool, type(None))
    if all(isinstance(value, scalar_types) for value in values):
        if all(value == values[0] for value in values[1:]):
            return values[0]
    return values


def _merge_result_dicts(dicts):
    keys = []
    seen = set()
    for item in dicts:
        for key in item:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return {
        key: _merge_gathered_values([item.get(key) for item in dicts])
        for key in keys
    }


def combine_dicts_across_gpus(local_dict):
    """Gather rank-local dictionaries and return one merged dictionary on all ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return local_dict
    world_size = dist.get_world_size()
    all_dicts = [None] * world_size
    dist.all_gather_object(all_dicts, local_dict)
    return _merge_result_dicts(all_dicts)


def sync_training_dicts_across_gpus(graph_dict, samples_dict):
    dict_list = [graph_dict, samples_dict]
    dist.broadcast_object_list(dict_list, src=0, device="cuda")
    return dict_list
