from pathlib import Path
import numpy as np
import shutil
import glob
import os

from umap import umap_

from torch_geometric.loader import DataLoader
import torch as torch
from torch import nn

from catalyst.src.ml.nn.models.alignn import Encoder, Processor, Decoder, ALIGNN
from catalyst.src.characterization.sodas.model.sodas import SODAS
import catalyst.src.utilities.sampling as sampling
from catalyst.src.ml.ml import ML

def main(ml_parameters):
    # initialize ML class and load graph data
    ml = ML()
    ml.set_params(ml_parameters)
    graph_data = []
    take = 1
    for i,graph in enumerate(glob.glob(os.path.join(ml.parameters['io_dict']['data_dir'],'*.pt'))):
        if i % take == 0:
            graph_data.append(torch.load(graph))
    print('Found ',len(graph_data),' graphs')

# sodas: either perform sodas projection or load a previously projected set of data
    projected_data = None
    y_values = []
    if ml.parameters['characterization_dict']['run_characterization']:
        print('Performing graph projections...')
        if os.path.isdir(ml.parameters['io_dict']['projection_dir']):
            shutil.rmtree(ml.parameters['io_dict']['projection_dir'])
        os.mkdir(ml.parameters['io_dict']['projection_dir'])

        encoded_data = []
        gids = []
        follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(graph_data[0], 'x_ang') else ['x_atm', 'x_bnd']
        for data in graph_data:
            y_values.append(data.y.item())
            gids.append(data.gid)
            loader = DataLoader([data], batch_size=1, shuffle=False, follow_batch=follow_batch,
                                num_workers=ml_parameters['loader_dict']['num_workers'])
            encoded_data.append(ml.parameters['characterization_dict']['model'].generate_gnn_latent_space(loader=loader,
                                                                                                device=ml.parameters[
                                                                                                    'device_dict'][
                                                                                                    'device'])[0])
        encoded_data = np.array(encoded_data)
        ml.parameters['characterization_dict']['model'].fit_preprocess(data=encoded_data)
        ml.parameters['characterization_dict']['model'].fit_dim_red(data=encoded_data)
        projected_data =  ml.parameters['characterization_dict']['model'].project_data(data=encoded_data)
        ml.parameters['characterization_dict']['model'].clear_gpu_memory()
        stored_projections = dict(
            projections = projected_data,
            gids = gids
        )
        np.save(os.path.join(ml.parameters['io_dict']['projection_dir'], 'saved_projection_data.npy'), stored_projections)
    else:
        print('Loading graph projections...')
        # when loading projections, you must have saved the data in the dictionary format used in the ml.parameters['run_sodas_projection'] section
        # this stored the graphs and projections in the correct order, together, so that they can be indexed together
        # note, this will rewrite the data variable to ensure that data and projected_data are in the same order
        loaded_projections = np.load(os.path.join(ml.parameters['io_dict']['projection_dir'], 'saved_projection_data.npy'),allow_pickle=True)
        gids = loaded_projections.item().get('gids')
        check = 0
        for data in graph_data:
            if data.gid not in gids:
                check = 1
                break
        if check:
            print('Projected data does not match loaded graphs...killing run...')
            exit(0)
        projected_data = loaded_projections.item().get('projections')
    ml.parameters['io_dict']['samples_dir'] = os.path.join(ml_parameters['io_dict']['main_path'], 'sampled_data')
    if os.path.isdir(ml.parameters['io_dict']['samples_dir']):
        shutil.rmtree(ml.parameters['io_dict']['samples_dir'])
    os.mkdir(ml.parameters['io_dict']['samples_dir'])

    # remove test data
    rng = np.random.default_rng(seed=ml.parameters['sampling_dict']['sampling_seed'])
    #note that y values are added here as the test set is sampled via y_binning
    test_idx, nontest_idx = sampling.run_sampling(graph_data,sampling_type=ml.parameters['sampling_dict']['sampling_types'][0],
                                                           split=ml.parameters['sampling_dict']['split'][0],rng=rng,
                                                           params_group=ml.parameters['sampling_dict']['params_groups'][0],y=y_values)
    #data = [graph_data[index] for index in nontest_idx]
    stored_test_data = dict(
        projections=[projected_data[index] for index in test_idx],
        gids=[graph_data[index].gid for index in test_idx]
    )
    projected_data = [projected_data[index] for index in nontest_idx]
    graph_data = [graph_data[index] for index in nontest_idx]
    np.save(os.path.join(ml.parameters['io_dict']['samples_dir'], 'stored_test_data.npy'),stored_test_data)

    # run pretraining
    pretraining_data = None
    if ml.parameters['model_dict']['pre_training']:
        # perform pretraining
        if os.path.isdir(ml.parameters['io_dict']['pretrain_dir']):
            shutil.rmtree(ml.parameters['io_dict']['pretrain_dir'])
        os.mkdir(ml.parameters['io_dict']['pretrain_dir'])
        # remove pretrain data
        pretrain_idx, nonpretrain_idx = sampling.run_sampling(projected_data, sampling_type=ml.parameters['sampling_dict']['sampling_types'][1],split=ml.parameters['sampling_dict']['split'][1],rng=rng,params_group=ml.parameters['sampling_dict']['params_groups'][1])
        #data = [graph_data[index] for index in nonpretrain_idx]
        stored_pretrain_data = dict(
            training_projections=[projected_data[index] for index in pretrain_idx],
            validation_projections=None,
            training=[graph_data[index].gid for index in pretrain_idx],
            validation=None
        )
        projected_data = [projected_data[index] for index in nonpretrain_idx]
        graph_data = [graph_data[index] for index in nonpretrain_idx]
        np.save(os.path.join(ml.parameters['io_dict']['pretrain_dir'], 'train_valid_split.npy'), stored_pretrain_data, fix_imports=False)

    # loop over number of requested models
    ml.parameters['io_dict']['model_dir'] = os.path.join(ml.parameters['io_dict']['main_path'],'model_samples')
    if os.path.isdir(ml.parameters['io_dict']['model_dir']):
        shutil.rmtree(ml.parameters['io_dict']['model_dir'])
    os.mkdir(ml.parameters['io_dict']['model_dir'])
    for iteration in range(ml.parameters['model_dict']['n_models']):
        ml.parameters['io_dict']['model_save_dir'] = os.path.join(ml.parameters['io_dict']['model_dir'], str(iteration))
        if os.path.isdir(ml.parameters['io_dict']['model_save_dir']):
            shutil.rmtree(ml.parameters['io_dict']['model_save_dir'])
        os.mkdir(ml.parameters['io_dict']['model_save_dir'])
        
        # sample data and train model
        train_idx, valid_idx = sampling.run_sampling(projected_data,sampling_type=ml.parameters['sampling_dict']['sampling_types'][2],
                                                           split=ml.parameters['sampling_dict']['split'][2],rng=rng,
                                                           params_group=ml.parameters['sampling_dict']['params_groups'][2])
        train_data = [graph_data[index].gid for index in train_idx]
        valid_data = [graph_data[index].gid for index in valid_idx]
        print('Using the remaining ',len(valid_data),' for validation')
        partitioned_data = dict(
            training_projections=[projected_data[index] for index in train_idx],
            validation_projections=[projected_data[index] for index in valid_idx],
            training=train_data,
            validation=valid_data
        )
        np.save(os.path.join(ml.parameters['io_dict']['model_save_dir'],'train_valid_split.npy'), partitioned_data)
         
if __name__ == "__main__":
    import time
    start_time = time.time()

    # setup parameters
    path = str(Path(__file__).parent)
    ml_parameters = dict(
        device_dict=dict(
            device='cuda',
            pin_memory=False,
        ),
        io_dict=dict(
            main_path=path,
            data_dir=os.path.join(path, 'graph_data'),
            model_dir='',
            model_save_dir='',
            pretrain_dir=os.path.join(path, 'pre_training'),
            samples_dir='',
            projection_dir=os.path.join(path, 'sodas_projection'),
        ),
        sampling_dict=dict(sampling_types=['y_bin','gaussian_mixture','gaussian_mixture'],
                           split=[0.2, 0.2, 0.9],
                           sampling_seed=112358,
                           params_groups=[{
                               'clusters': 5,
                           },{
                               'clusters': 5,
                           },{
                               'clusters': 5,
                            }]
                           ),
        loader_dict=dict(
            num_workers=0,
        ),
        model_dict=dict(
            n_models=2,
            pre_training=True
        ),
        characterization_dict=dict(
            run_characterization=True,
            model=SODAS(mod=ALIGNN(
                encoder=Encoder(num_species=119, cutoff=4.0, dim=100, act_func=nn.SiLU()),
                processor=Processor(num_convs=5, dim=100),
                decoder=Decoder(in_dim=100, out_dim=10, act_func=nn.SiLU())
            ),
                ls_mod=umap_.UMAP(n_neighbors=15, min_dist=0.5, n_components=2)
            ),
        )
    )

    main(ml_parameters)

    print("--- %s seconds ---" % (time.time() - start_time))




