from torch.distributed import init_process_group,destroy_process_group
import torch.multiprocessing as mp
import torch.distributed as dist
from numba import cuda
import torch
import gc
import os

def set_spawn_method(parameters):
    if 'ddp_backend' in parameters:
        if 'ddp_backend' == 'nccl':
            mp.set_start_method('spawn')

def ddp_setup(rank: int ,world_size ,backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend=backend, rank=rank, world_size=world_size,init_method="env://?use_libuv=False")

def cuda_destroy():
    # collect memory via garbage collection
    gc.collect()
    # loop through active devices (assumes you are using all devices available), clear their memory, and then reset the device and close cuda
    for gpu_id in range(torch.cuda.device_count()):
        cuda.select_device(gpu_id)
        torch.cuda.empty_cache()
        device = cuda.get_current_device()
        device.reset()
        cuda.close()

def ddp_destroy():
    dist.barrier()
    destroy_process_group()

def reduce_tensor(tensor):
    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    return rt

def combine_dicts_across_gpus(local_dict):
    world_size = dist.get_world_size()
    all_dicts = [None] * world_size
    dist.all_gather_object(all_dicts, local_dict)
    return all_dicts

def sync_training_dicts_across_gpus(graph_dict, samples_dict):
    dict_list = [graph_dict,samples_dict]
    dist.broadcast_object_list(dict_list, src=0, device='cuda')
    return dict_list


