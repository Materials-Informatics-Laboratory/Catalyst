from pathlib import Path
import numpy as np
import shutil
import glob
import sys
import os

from torch_geometric.loader import DataLoader
from umap import umap_
import torch as torch

import catalyst.src.utilities.sampling as sampling
from catalyst.src.ml.training import run_training, run_pre_training
from catalyst.src.ml.ml import ML

from catalyst.src.sodas.nn.models.alignn import Encoder, Processor, Decoder, ALIGNN
from catalyst.src.sodas.model.sodas import SODAS

path = str(Path(__file__).parent)

# setup parameters
ml_parameters = dict(gnn_dim = 10,
                        num_convs = 5,
                        num_inputs = 5,
                        num_epochs = 2,
                        BATCH_SIZE = 1,
                        n_models = 3,
                        graph_cutoff = 4.0,
                        LEARN_RATE = 2e-4,
                        train_tolerance = 1e-5,
                        is_dihedral = False,
                        remove_old_model = False,
                        interpretable = False,
                        pre_training = False,
                        run_pretrain = False,
                        write_indv_pred = False,
                        restart_training = True,
                        sodas_projection = True,
                        run_sodas_projection = True,
                        device = 'cpu',
                        restart_model_name = r'C:\Users\jc112358\PycharmProjects\pythonProject\venv\Lib\site-packages\remi\examples\train_model\models\0\model_2024-03-29_11-47-09',
                        graph_data_dir = os.path.join(path,'data'),
                        model_dir=None,
                        pretrain_dir=os.path.join(path,'pre_training'),
                        results_dir = None,
                        samples_dir = None,
                        elements=['Co','Mo','Cu','Ni','Fe'],
                        sampling_dict=dict(test_sampling_type = 'y_bin',
                                            pretraining_sampling_type='kmeans',
                                            sampling_type='gaussian_mixture',
                                            split=[0.1,0.25,0.5,0.5], # [test,pretrain,train,val], test is percent of total data left out of training and validation, pretrain is selected and not replaced for pretraining, the remaining data is split into the train vs val split
                                            clusters=3
                        ),
                        sodas_dict = dict(
                            sodas_model=SODAS(mod=ALIGNN(
                                encoder=Encoder(num_species=5, cutoff=4.0, dim=10),
                                processor=Processor(num_convs=5, dim=10),
                                decoder=Decoder(node_dim=10, out_dim=10)
                                ),
                                ls_mod=umap_.UMAP(n_neighbors=20, min_dist=0.1, n_components=3)
                            ),
                            projection_dir=os.path.join(path,'sodas_projection')
                        )
                    )

if ml_parameters['restart_training']:
    wp = ml_parameters['restart_model_name']
    if sys.platform.startswith('win'):
        print(wp)
    else:
        ml.parameters['model_dir'] = save_dir
    exit(0)
else:
    model_dir = os.path.join(path,'models')
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

# initialize ML class and load graph data
ml = ML()
ml.set_params(ml_parameters, model_dir)
data = []
for graph in glob.glob(os.path.join(ml.parameters['graph_data_dir'],'*.pt')):
    data.append(torch.load(graph))

# sodas: either perform sodas projection or load a previously projected set of data
projected_data = None
if ml.parameters['run_sodas_projection']:
    if os.path.isdir(ml.parameters['sodas_dict']['projection_dir']):
        shutil.rmtree(ml.parameters['sodas_dict']['projection_dir'])
    os.mkdir(ml.parameters['sodas_dict']['projection_dir'])
    follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(data[0], 'x_ang') else ['x_atm']
    loader = DataLoader(data, batch_size=1, shuffle=False, follow_batch=follow_batch)
    encoded_data = ml.parameters['sodas_dict']['sodas_model'].generate_gnn_latent_space(loader=loader, device=ml.parameters['device'])
    ml.parameters['sodas_dict']['sodas_model'].fit_preprocess(data=encoded_data)
    ml.parameters['sodas_dict']['sodas_model'].fit_dim_red(data=encoded_data)
    projected_data =  ml.parameters['sodas_dict']['sodas_model'].project_data(data=encoded_data)

    stored_projection = dict(
        projections = projected_data,
        graphs = data
    )
    np.save(os.path.join(ml.parameters['sodas_dict']['projection_dir'], 'saved_projection_data.npy'), stored_projection)
elif ml.parameters['sodas_projection']:
    # when loading projections, you must have saved the data in the dictionary format used in the ml.parameters['run_sodas_projection'] section
    # this stored the graphs and projections in the correct order, together, so that they can be indexed together
    # note, this will rewrite the data variable to ensure that data and projected_data are in the same order
    loaded_projections = np.load(os.path.join(ml.parameters['sodas_dict']['projection_dir'], 'saved_projection_data.npy'),allow_pickle=True)
    data = loaded_projections.item().get('graphs')
    projected_data = loaded_projections.item().get('projections')

# remove validation data
test_idx, nontest_idx = sampling.run_sampling(data,sampling_type=ml.parameters['sampling_dict']['test_sampling_type'],
                                                       split=ml.parameters['sampling_dict']['split'][0],
                                                       nclusters=ml.parameters['sampling_dict']['clusters'])
test_data = [data[index] for index in test_idx]
data = [data[index] for index in nontest_idx]
projected_data_test = [projected_data[index] for index in test_idx]
projected_data = [projected_data[index] for index in nontest_idx]

# run pretraining
pretraining_data = None
if ml.parameters['run_pretrain']:
    # remove pretrain data
    pretrain_idx, nonpretrain_idx = sampling.run_sampling(projected_data, sampling_type=ml.parameters['sampling_dict'][
        'pretraining_sampling_type'],
                                                  split=ml.parameters['sampling_dict']['split'][1],
                                                  nclusters=ml.parameters['sampling_dict']['clusters'])
    pretraining_data = [data[index] for index in pretrain_idx]
    data = [data[index] for index in nonpretrain_idx]
    projected_data_pretrain = [projected_data[index] for index in pretrain_idx]
    projected_data = [projected_data[index] for index in nonpretrain_idx]

    # perform pretraining
    if os.path.isdir(ml.parameters['pretrain_dir']):
        shutil.rmtree(ml.parameters['pretrain_dir'])
    os.mkdir(ml.parameters['pretrain_dir'])
    ml.set_model()
    partitioned_data = dict(training=pretraining_data,
                                validation=None)
    np.save(os.path.join(ml.parameters['pretrain_dir'], 'train_valid_split.npy'), partitioned_data)
    run_pre_training(data=partitioned_data, parameters=ml.parameters, model=ml.model)

# save different bins of data
ml.parameters['samples_dir'] = os.path.join(path,'sampled_data')
if os.path.isdir(ml.parameters['samples_dir']):
    shutil.rmtree(ml.parameters['samples_dir'])
os.mkdir(ml.parameters['samples_dir'])
np.save(os.path.join(ml.parameters['samples_dir'], 'test_data.npy'), test_data)
np.save(os.path.join(ml.parameters['samples_dir'], 'test_data_projected.npy'), projected_data_test)
np.save(os.path.join(ml.parameters['samples_dir'], 'remaining_data.npy'), data)
np.save(os.path.join(ml.parameters['samples_dir'], 'remaining_data_projected.npy'), projected_data)
if ml.parameters['run_pretrain']:
    np.save(os.path.join(ml.parameters['samples_dir'], 'pretraining_data.npy'), pretraining_data)
    np.save(os.path.join(ml.parameters['samples_dir'], 'pretraining_data_projected.npy'), projected_data_pretrain)

# reload model if performing model restart
if ml_parameters['restart_training']:
    ml.model = ml.model.to(ml.parameters['device'])
    ml.model.load_state_dict(torch.load(ml_parameters['restart_model_name']))

# loop over number of requested models
for iteration in range(ml.parameters['n_models']):
    if ml_parameters['restart_training']:
        print('Restarting model training using model ',ml_parameters['restart_model_name'])


    else:
        print('Training model ',iteration)
        save_dir = os.path.join(model_dir,str(iteration))
        os.mkdir(save_dir)
        ml.parameters['model_dir'] = save_dir

    # if pretraining is on this will load the pretrained model, if not it will initialize a new model
    if ml.parameters['pre_training'] == False:
        ml = ML()
        ml.set_params(ml_parameters,model_dir)
        ml.set_model()
    else:
        ml.set_model()
        ml.model = ml.model.to(ml.parameters['device'])
        ml.model.load_state_dict(torch.load(os.path.join(ml.parameters['pretrain_dir'],'model_pre')))

    # sample data and train model
    train_idx, valid_idx = sampling.run_sampling(projected_data,sampling_type=ml.parameters['sampling_dict']['sampling_type'],
                                                   split=ml.parameters['sampling_dict']['split'][2],
                                                   nclusters=ml.parameters['sampling_dict']['clusters'])
    train_data = [data[index] for index in train_idx]
    valid_data = [data[index] for index in valid_idx]
    partitioned_data = dict(training=train_data,
                            validation=valid_data)
    np.save(os.path.join(save_dir,'train_valid_split.npy'), partitioned_data)
    run_training(data=partitioned_data,parameters=ml.parameters,model=ml.model)






