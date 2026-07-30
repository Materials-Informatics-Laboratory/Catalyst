from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import gc
import os


def set_spawn_method(parameters):
    """Use the spawn multiprocessing context for NCCL-backed DDP runs.

    ``parameters`` is normally the full Catalyst parameter dictionary, where
    the backend lives under ``device_dict``.  A flat mapping is also accepted
    for backward compatibility.
    """
    device_dict = parameters.get("device_dict", parameters)
    backend = str(device_dict.get("ddp_backend", "")).lower()

    if backend == "nccl":
        mp.set_start_method("spawn", force=True)


def ddp_setup(rank: int, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method="env://?use_libuv=False",
    )


def ddp_model(model, find_unused_parameters, rank, batchnorm):
    # 1) Set device for this rank and move model there
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    model.to(device)

    # 2) Convert BN -> SyncBatchNorm BEFORE wrapping in DDP
    if batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 3) Wrap in DDP on this device
    model = DDP(
        model,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=find_unused_parameters,
    )
    return model


def cuda_destroy():
    """Release Python, CUDA, and initialized process-group resources."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ddp_destroy():
    """Synchronize and destroy DDP only when a group is initialized."""
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        finally:
            destroy_process_group()


def reduce_tensor(tensor):
    """Average a tensor across the active distributed process group."""
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size() if dist.is_initialized() else 1
    return rt


def combine_dicts_across_gpus(local_dict):
    world_size = dist.get_world_size()
    all_dicts = [None] * world_size
    dist.all_gather_object(all_dicts, local_dict)
    return all_dicts


def sync_training_dicts_across_gpus(graph_dict, samples_dict):
    dict_list = [graph_dict, samples_dict]
    dist.broadcast_object_list(dict_list, src=0, device="cuda")
    return dict_list
