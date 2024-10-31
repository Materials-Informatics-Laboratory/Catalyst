from ..graph.graph import Atomic_Graph_Data, Graph_Data
from pathlib import PurePath
import numpy as np
import torch
import glob
import os

import platform
import psutil
import GPUtil

def write_labelled_extxyz(filename,labels,atoms,cutoff):
    of = open(filename, 'w')
    of.write(str(len(labels)) + '\n')
    of.write('Lattice="')
    for cd in atoms.get_cell():
        for c in cd:
            of.write(str(c) + ' ')
    of.write(
        ' Properties=species:S:1:pos:R:3:y:R:1 cutoff ' + str(cutoff) + ' pbc="T T T"\n')
    for j, atom in enumerate(atoms):
        of.write(atom.symbol + ' ')
        for pos in atom.position:
            of.write(str(pos) + ' ')
        of.write(str(labels[j].item()) + '\n')
    of.close()

def get_system_info():
    system_info = platform.uname()
    memory_info = psutil.virtual_memory()
    gpus = GPUtil.getGPUs()

    if not gpus:
        info = dict(
            system=system_info.system,
            node=system_info.node,
            release=system_info.release,
            version=system_info.version,
            machine=system_info.machine,
            processor=system_info.processor,
            cpu_count=psutil.cpu_count(logical=False),
            logical_count=psutil.cpu_count(logical=True),
            total_memory=memory_info.total,
            ngpus=len(gpus),
        )
    else:
        info = dict(
            system=system_info.system,
            node=system_info.node,
            release=system_info.release,
            version=system_info.version,
            machine=system_info.machine,
            processor=system_info.processor,
            cpu_count=psutil.cpu_count(logical=False),
            logical_count=psutil.cpu_count(logical=True),
            total_memory=memory_info.total,
            ngpus=len(gpus),
            gpu_type=gpus[0].name,
            gpu_driver=gpus[0].driver,
            gpu_memory=gpus[0].memoryTotal
        )
    return info

def read_training_data(params,samples_file,pretrain=False,format=0):
    training_graphs = None
    training_samples = None
    validation_graphs = None
    validation_samples = None

    graph_files = glob.glob(os.path.join(params['io_dict']['data_dir'],'*'))
    samples = np.load(samples_file, allow_pickle=True)
    training_samples = samples.item().get('training')
    #unique_samples = np.unique(np.array(training_samples))
    if pretrain == False:
        validation_samples = samples.item().get('validation')

    if format == 0:
        gids = [PurePath(graph).parts[-1].split('.')[0] for graph in graph_files]
        if len(gids) == 0:
            print('Error: no graph files found...')
            exit(0)
    else:
        gids = [torch.load(gname)['gid'] for gname in graph_files]

    cross_list = set(training_samples).intersection(gids)
    idx = [gids.index(x) for x in cross_list]
    selected_graphs = [graph_files[i] for i in idx]
    if format == 0:
        training_graphs = [torch.load(x) for x in selected_graphs]
    else:
        training_graphs = [x for x in selected_graphs]
    if pretrain == False:
        cross_list = set(validation_samples).intersection(gids)
        idx = [gids.index(x) for x in cross_list]
        selected_graphs = [graph_files[i] for i in idx]
        if format == 0:
            validation_graphs = [torch.load(x) for x in selected_graphs]
        else:
            validation_graphs = [x for x in selected_graphs]
    return dict(training=training_graphs, validation=validation_graphs), dict(training_samples=training_samples,validation_samples=validation_samples)

def port_graphdata_to_atomicgraphdata(path):
    graph_files = glob.glob(os.path.join(path, '*'))
    for graph in graph_files:
        data = torch.load(graph)
        if isinstance(data,Graph_Data):
            if not hasattr(data, 'atoms'):
                atoms = None
            else:
                atoms = data.atoms
            if not hasattr(data, 'y'):
                y = None
            else:
                y = data.y
            if not hasattr(data, 'x_ang'):
                x_ang = None
            else:
                x_ang = data.x_ang
            if not hasattr(data, 'mask_dih_ang'):
                mask_dih_ang = None
            else:
                mask_dih_ang = data.mask_dih_ang
            if not hasattr(data, 'ang_amounts'):
                ang_amounts = None
            else:
                ang_amounts = data.ang_amounts
            new_data = Atomic_Graph_Data(
                atoms=atoms,
                edge_index_G=data.edge_index_G,
                edge_index_A=data.edge_index_A,
                x_atm=data.x_atm,
                x_bnd=data.x_bnd,
                x_ang=x_ang,
                mask_dih_ang=mask_dih_ang,
                atm_amounts=data.atm_amounts,
                bnd_amounts=data.bnd_amounts,
                ang_amounts=ang_amounts,
            )
            new_data.y = y
            if not hasattr(data, 'gid'):
                new_data.generate_gid()
            else:
                new_data.gid = data.gid
            os.remove(graph)
            torch.save(new_data, os.path.join(path, new_data.gid + '.pt'))

def port_graphs_without_gids(params):
    graph_files = glob.glob(os.path.join(params['data_dir'], '*'))

    check = 1
    for graph in graph_files:
        data = torch.load(graph)
        if not hasattr(data, 'gid'):
            if check:
                from ..graph.graph import Atomic_Graph_Data
                check = 0

            if not hasattr(data, 'atoms'):
                atoms = None
            else:
                atoms = data.atoms
            if not hasattr(data, 'y'):
                y = None
            else:
                y = data.y
            if not hasattr(data, 'x_ang'):
                x_ang = None
            else:
                x_ang = data.x_ang
            if not hasattr(data, 'mask_dih_ang'):
                mask_dih_ang = None
            else:
                mask_dih_ang = data.mask_dih_ang
            if not hasattr(data, 'ang_amounts'):
                ang_amounts = None
            else:
                ang_amounts = data.ang_amounts

            new_data = Atomic_Graph_Data(
                atoms=atoms,
                edge_index_G=data.edge_index_G,
                edge_index_A=data.edge_index_A,
                x_atm=data.x_atm,
                x_bnd=data.x_bnd,
                x_ang=x_ang,
                mask_dih_ang=mask_dih_ang,
                atm_amounts=data.atm_amounts,
                bnd_amounts=data.bnd_amounts,
                ang_amounts=ang_amounts,
            )
            new_data.y = y
            data = new_data
            data.generate_gid()
        elif data.gid is None:
            data.generate_gid()
        os.remove(graph)
        torch.save(data, os.path.join(params['data_dir'], data.gid + '.pt'))
