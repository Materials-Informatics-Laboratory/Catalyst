from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from .....graph.graph import Atomic_Graph_Data, Generic_Graph_Data, Graph_Data
import math


def _shared_sampler_seed(parameters):
    """Return one deterministic seed shared by every DDP rank.

    DistributedSampler assumes that all ranks construct the same global
    permutation and then take disjoint slices.  Using rank-local random seeds
    breaks that contract and can duplicate/omit training examples.
    """
    sampling = parameters.get("sampling_dict", {}) or {}
    return int(sampling.get("sampling_seed", 0))


def setup_dataloader(data, cat, loader_params):
    parameters = cat.parameters
    if data is None or len(data) == 0:
        raise ValueError("Cannot create a DataLoader from an empty dataset.")

    follow_batch = []
    if isinstance(data[0], Atomic_Graph_Data):
        follow_batch = ["x_atm", "x_bnd", "x_ang"] if getattr(data[0], "x_ang", None) is not None else ["x_atm", "x_bnd"]
    elif isinstance(data[0], Generic_Graph_Data):
        follow_batch = ["node_G", "node_A", "edge_A"] if getattr(data[0], "edge_A", None) is not None else ["node_G", "node_A"]
    elif isinstance(data[0], Graph_Data):
        follow_batch = []

    if parameters["device_dict"]["run_ddp"]:
        sampler = DistributedSampler(
            data,
            shuffle=bool(loader_params["shuffle"]),
            seed=_shared_sampler_seed(parameters),
        )
        sampler.set_epoch(int(loader_params.get("epoch", 0)))
        return DataLoader(
            data,
            batch_size=math.ceil(
                loader_params["batch_size"] / parameters["device_dict"]["world_size"]
            ),
            pin_memory=parameters["device_dict"]["pin_memory"],
            follow_batch=follow_batch,
            sampler=sampler,
            num_workers=parameters["loader_dict"]["num_workers"],
        )

    return DataLoader(
        data,
        pin_memory=parameters["device_dict"]["pin_memory"],
        batch_size=loader_params["batch_size"],
        shuffle=loader_params["shuffle"],
        follow_batch=follow_batch,
        num_workers=parameters["loader_dict"].get("num_workers", 0),
    )
