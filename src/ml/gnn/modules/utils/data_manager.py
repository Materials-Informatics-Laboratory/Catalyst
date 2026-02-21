from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from .....graph.graph import Atomic_Graph_Data, Generic_Graph_Data, Graph_Data

def setup_dataloader(data,cat,loader_params):
    parameters = cat.parameters
    loader = None
    if isinstance(data[0], Atomic_Graph_Data):
        follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(data[0], 'x_ang') else ['x_atm', 'x_bnd']
    elif isinstance(data[0], Generic_Graph_Data):
        follow_batch = ['node_G', 'node_A', 'edge_A'] if hasattr(data[0], 'edge_A') else ['node_G','node_A']

    if parameters['device_dict']['run_ddp']:
        sampler = DistributedSampler(data,shuffle=loader_params['shuffle'],
                                               seed=random.randint(-sys.maxsize - 1, sys.maxsize))
        sampler.set_epoch(loader_params['epoch'] )
        loader= DataLoader(data, batch_size=math.ceil(
            loader_params['batch_size'] / parameters['device_dict']['world_size']),
                                      pin_memory=parameters['device_dict']['pin_memory'],
                                      follow_batch=follow_batch,
                                      sampler=sampler,
                                      num_workers=parameters['loader_dict']['num_workers'])
    else:
        loader = DataLoader(data, pin_memory=parameters['device_dict']['pin_memory'],
                                      batch_size=loader_params['batch_size'] ,
                                      shuffle=loader_params['shuffle'],
                                      follow_batch=follow_batch)
    return loader
