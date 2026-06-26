import torch

from .gnn.modules.utils.data_manager import setup_dataloader
from ..data.utils import save_dictionary
from ..data.model_data import save_model


from .utils.distributed import ddp_destroy, ddp_setup, reduce_tensor, ddp_model
from ..utilities.sampling import active_sampling, random_
from .utils.optimizer import set_optimizer
from .utils.memory import optimizer_to
from .utils.loss import loss_setup,active_learning_setup

import shutil
import glob
import time
import sys
import os


def setup_training(rank,cat=None):
    if rank == 0:
        print('Training model...')

    parameters = cat.parameters

    parameters['io_dict']['model_dir'] = None
    del parameters['io_dict']['model_dir']
    parameters['io_dict']['model_dir'] = os.path.join(parameters['io_dict']['main_path'], 'models',
                                                      'training')
    if rank == 0:
        if os.path.isdir(parameters['io_dict']['model_dir']):
            shutil.rmtree(parameters['io_dict']['model_dir'])
        os.makedirs(parameters['io_dict']['model_dir'], exist_ok=True)

    parameters['model_dict']['model'].load_data(parameters,
                samples_file=os.path.join(parameters['io_dict']['samples_dir'],'train_valid_split.npy'),
                                                         format=parameters['io_dict']['graph_read_format'], rank=rank)

    if parameters['device_dict']['run_ddp']:
        ddp_setup(rank, parameters['device_dict']['world_size'], parameters['device_dict']['ddp_backend'])

def run_active_learning(rank,cat=None):
    # set up run
    epoch_times = []
    running_valid_delta = []
    L_train, L_valid = [], []
    total_active_samples = []
    min_loss_train = 1.0E30
    min_loss_valid = 1.0E30
    shuffle_counter = 0
    met_tolerance = 0
    iteration = 0
    parameters = cat.parameters
    best_model_state = None
    best_optimizer_state = None
    reset_optimizer = False

    if parameters['device_dict']['run_ddp']:
        ddp_setup(rank, parameters['device_dict']['world_size'], parameters['device_dict']['ddp_backend'])

    parameters['io_dict']['model_dir'] = None
    del parameters['io_dict']['model_dir']
    parameters['io_dict']['model_dir'] = os.path.join(parameters['io_dict']['main_path'], 'models','active_learning')
    if rank == 0:
        if os.path.isdir(parameters['io_dict']['model_dir']):
            shutil.rmtree(parameters['io_dict']['model_dir'])
        os.makedirs(parameters['io_dict']['model_dir'], exist_ok=True)

    model, model_data = setup_model(cat, rank=rank, load=True)
    active_loss_dict = active_learning_setup(parameters,{'model':model,'metadata':model_data})

    if parameters['io_dict']['graph_read_format'] != 2:
        training_graphs = read_graphs_from_gids(parameters['model_dict']['active_learning_params_group']['training_data_dir'],
                                                model_data['run_information']['samples']['training_samples'])
    else:
        training_graphs = load_dictionary(glob.glob(os.path.join(parameters['model_dict']['active_learning_params_group']['training_data_dir'], 'graphs.data'))[0])['graphs']
    new_graphs = read_graphs_from_gids(
        parameters['io_dict']['data_dir'])

    parameters['model_dict']['active_learning_params_group']['sampling_params_group']['data'] = new_graphs
    parameters['model_dict']['active_learning_params_group']['sampling_params_group']['training_data'] = training_graphs
    parameters['model_dict']['active_learning_params_group']['sampling_params_group']['y'] = [graph.y for graph in new_graphs]
    parameters['model_dict']['active_learning_params_group']['sampling_params_group']['training_y'] = [graph.y for graph in training_graphs]

    # retrain model for a few epochs with new data added
    data = {
        'training': [],
        'validation': []
    }
    if parameters['model_dict']['active_learning_params_group']['training_params_group']['train_with_previous']:
        if isinstance(parameters['model_dict']['active_learning_params_group']['training_params_group']['percent_use_previous'],float):
            samples, non_samples = random_(training_graphs.copy(),
                                          parameters['model_dict']['active_learning_params_group']['training_params_group']['percent_use_previous'],
                                          rng=parameters['model_dict']['active_learning_params_group']['sampling_params_group']['rng'])
            data['training'] = [training_graphs[i] for i in samples]
        else:
            data['training'] = training_graphs.copy()

    loader_train = setup_dataloader({
        'training': training_graphs,
    }, cat=cat, mode=0)
    loss_train, train_info = test_non_intepretable_internal(loader_train, model, parameters, rank=rank,
                                                            return_test_info=True)

    print('Initial training loss: ',loss_train)

    while iteration < parameters['model_dict']['active_learning_params_group']['training_params_group']['iterations']:

        L_train.append([])
        L_valid.append([])
        print('Active learning iteration: ',iteration)
        # determine which point(s) to add to model
        new_samples, remaining_data = active_sampling(
            parameters['model_dict']['active_learning_params_group']['sampling_params_group'])
        total_active_samples.append(new_samples)
        loader_train = setup_dataloader({'training': new_graphs, }, cat=cat, mode=0)
        loss_train, train_info = test_non_intepretable_internal(loader_train, model, parameters, rank=rank,
                                                               return_test_info=True)

        data['training'].extend([new_graphs[i] for i in new_samples])
        data['validation'] = [new_graphs[i] for i in remaining_data]
        # add points to dataloader for model
        loader_train, loader_valid = setup_dataloader(data=data, cat=cat, mode=1)

        ep = 0
        patience_counter = 0
        patience = parameters['model_dict']['patience'] # epochs to wait before LR reduction
        if parameters['model_dict']['optimizer_params'].get('dynamic_lr'):
            lr_decay_factor = parameters['model_dict']['optimizer_params']['params_group']['lr_decay_factor']
        else:
            lr_decay_factor = 0.0
        worsen_tolerance = parameters['model_dict']['worsen_tolerance']  # 5% allowed
        while ep < parameters['model_dict']['active_learning_params_group']['training_params_group']['epochs_per_iteration']:
            if rank == 0:
                if ep > 0:
                    start_time = time.time()
                print('Epoch ', ep + 1, ' of ', parameters['model_dict']['active_learning_params_group']['training_params_group']['epochs_per_iteration'])
                sys.stdout.flush()

            if parameters['device_dict']['run_ddp']:
                parameters['model_dict']['optimizer_params']['params_group'][
                    'params'] = model.module.processor.parameters()
            else:
                parameters['model_dict']['optimizer_params']['params_group']['params'] = model.processor.parameters()

            # --- Data reshuffle ---
            if parameters['loader_dict']['shuffle_loader'] == True:
                if ep % parameters['loader_dict']['shuffle_steps'] == 0 and ep > 0:
                    if rank == 0:
                        print('Shuffling training data...')
                    loader_train, loader_valid = setup_dataloader(data=data, cat=cat, epoch=ep, reshuffle=True,
                                                                      mode=1)
                    shuffle_counter += 1

            # --- Optimizer setup ---
            if parameters['device_dict']['run_ddp']:
                parameters['model_dict']['optimizer_params']['params_group'][
                    'params'] = model.module.processor.parameters()
            else:
                parameters['model_dict']['optimizer_params']['params_group'][
                    'params'] = model.processor.parameters()

            if reset_optimizer:
                optimizer.load_state_dict(best_optimizer_state)
                reset_optimizer = False
            else:
                optimizer = set_optimizer(parameters)
            optimizer_to(optimizer, parameters['device_dict']['device'])
            # --- Learning rate scheduling on plateau ---
            if patience_counter >= patience:
                if rank == 0:
                    print(f"No improvement for {patience} epochs. Reducing LR by factor {lr_decay_factor}.")
                if lr_decay_factor > 0.0:
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] * lr_decay_factor
                patience_counter = 0

            # --- Train & Validate ---
            training_dict = {
                'params':parameters,
            }
            loss_train = train(model,optimizer,loader,training_dict)
            loss_valid = test_non_intepretable_internal(loader_valid, model, parameters, rank=rank)
            L_train[-1].append(loss_train)
            L_valid[-1].append(loss_valid)
            if rank == 0:
                if ep > 0:
                    epoch_times.append(time.time() - start_time)
                    print('epoch_time = ', time.time() - start_time, ' seconds',
                          ' Average epoch time = ', sum(epoch_times) / float(len(epoch_times)), ' seconds')
                print('Train loss = ', loss_train, ' Validation loss = ', loss_valid)

                # --- Save best model by validation only ---
                if loss_valid < min_loss_valid:
                    min_loss_valid = loss_valid
                    check = 0
                    if parameters['model_dict'].get('strict_loss_policy'):
                        if loss_train < min_loss_train:
                            min_loss_train = loss_train
                            check = 1
                    else:
                        check = 1
                    if check:
                        if rank == 0:
                            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                            best_optimizer_state = optimizer.state_dict()
                            model_params_group = {
                                'previous_samples_added': samples,
                                'new_samples':new_samples,
                                'all_actie_samples':total_active_samples,
                                'data_loader': loader_train,
                                'L_train': L_train[-1][-1],
                                'L_valid': L_valid[-1][-1],
                                'iteration':iteration,
                            }
                            save_model(model=model, cat=cat, model_params_group=model_params_group,
                                       remove_old_models=parameters['io_dict']['remove_old_model'])
                        patience_counter = 0
                else:
                    patience_counter += 1

                    # --- Revert if validation worsens too much ---
                    if loss_valid > min_loss_valid * worsen_tolerance and best_model_state is not None:
                        if rank == 0:
                            print(
                                f"Validation worsened by more than {100 * worsen_tolerance:.1f}%. Reverting to best model.")
                        model.load_state_dict(best_model_state)
                        model.to(parameters['device_dict']['device'])
                        reset_optimizer = True

            if ep > 0:
                if rank == 0 and parameters['io_dict']['training_info_nwrite_steps'] % ep == 0:
                    print('Writing run information...')
                    run_data = {
                        'epoch_timings': epoch_times,
                        'times_loader_shuffled': shuffle_counter,
                        'met_tolerance': met_tolerance,
                        'training_loss': L_train,
                        'validation_loss': L_valid,
                    }
                    save_dictionary(fname=os.path.join(cat.parameters['io_dict']['model_dir'], 'run_information.npy'),
                                    data=run_data)

            ep += 1

        loss_valid, test_info = test_non_intepretable_internal(loader_valid, model, parameters, rank=rank,
                                                               return_test_info=True)
        loss_train, train_info = test_non_intepretable_internal(loader_train, model, parameters, rank=rank,
                                                                return_test_info=True)
        print('Final losses for iteration ',iteration,' training loss: ',loss_train,' validation loss: ',loss_valid)

        new_graphs = [new_graphs[i] for i in remaining_data]
        parameters['model_dict']['active_learning_params_group']['sampling_params_group']['data'] = new_graphs
        parameters['model_dict']['active_learning_params_group']['sampling_params_group']['y'] = [graph.y for graph in
                                                                                                  new_graphs]

        iteration += 1

def run_training(rank,cat=None):
    parameters = cat.parameters
    epoch_times = []
    running_valid_delta = []
    L_train, L_valid = [], []
    shuffle_counter = 0
    met_tolerance = 0
    patience_counter = 0
    ep = 0
    min_loss_train = 1.0E30
    min_loss_valid = 1.0E30
    best_model_state = None
    best_optimizer_state = None
    reset_optimizer = False

    patience = parameters['model_dict']['patience']
    worsen_tolerance = parameters['model_dict']['worsen_tolerance']

    setup_training(rank=rank,cat=cat)

    model = parameters['model_dict']['model']
    model.device = parameters['device_dict']['device']
    if parameters['device_dict']['run_ddp']:
        model.model = ddp_model(model=model.model,
                      find_unused_parameters=parameters['device_dict']['find_unused_parameters'],
                      rank=rank, batchnorm=parameters['model_dict']['batchnorm'])
    else:
        #model.compile_model()
        pass
    model.set_optimizer_(parameters=parameters)
    #model.load_training_data(parameters, os.path.join(parameters['io_dict']['samples_dir'], 'train_valid_split.npy'),
    #                         format=parameters['io_dict']['graph_read_format'], rank=rank)
    model.set_dataloader(cat=cat, epoch=ep)


    while ep < parameters['model_dict']['num_epochs']:
        if rank == 0:
            if ep > 0:
                start_time = time.time()
            print('Epoch ', ep + 1, ' of ', parameters['model_dict']['num_epochs'])
            sys.stdout.flush()
        if parameters['device_dict']['run_ddp']:
            model.training_loader.sampler.set_epoch(ep)
            model.validation_loader.sampler.set_epoch(ep)


        # --- Train & Validate ---
        training_dict = {
            'params':parameters,
        }
        loss_train = model.train(training_dict=training_dict)
        loss_valid = model.validate(parameters=parameters,rank=rank)

        if rank == 0:
            if ep > 0:
                epoch_times.append(time.time() - start_time)
                print('epoch_time = ', time.time() - start_time, ' seconds',
                      ' Average epoch time = ', sum(epoch_times) / float(len(epoch_times)), ' seconds')
            print('Train loss = ', loss_train, ' Validation loss = ', loss_valid)
        L_train.append(loss_train)
        L_valid.append(loss_valid)

        # --- Save best model by validation only ---
        if loss_valid < min_loss_valid:
            min_loss_valid = loss_valid
            if parameters['model_dict'].get('strict_loss_policy'):
                if loss_train < min_loss_train:
                    min_loss_train = loss_train
                    check = 1
            else:
                check = 1
                min_loss_train = loss_train
            if check:
                if rank == 0:
                    print('Saving model checkpoint...')
                    model.save_checkpoint(parameters, ep, rank=rank)
                patience_counter = 0  # reset patience
        else:
            patience_counter += 1

            # --- Revert if validation worsens too much ---
            if loss_valid > min_loss_valid * worsen_tolerance and best_model_state is not None:
                if rank == 0:
                    print(f"Validation worsened by more than {100 * worsen_tolerance:.1f}%. Reverting to best model.")
                ep = trainer.load_checkpoint()

        # --- Convergence check using deltas ---
        if ep > 1:
            delta_val = loss_valid - L_valid[-2]
            running_valid_delta.append(abs(delta_val))
            if len(running_valid_delta) > parameters['model_dict']['max_deltas']:
                running_valid_delta.pop(0)
                if rank == 0:
                    print('Running validation delta = ',
                          sum(running_valid_delta) / len(running_valid_delta))
            if len(running_valid_delta) == parameters['model_dict']['max_deltas']:
                if (sum(running_valid_delta) / len(running_valid_delta) < parameters['model_dict']['train_delta']
                        and (sum(L_valid[-parameters['model_dict']['max_deltas']:]) / parameters['model_dict'][
                            'max_deltas']) < parameters['model_dict']['train_tolerance']):
                    if rank == 0:
                        print('Validation delta satisfies set tolerance...exiting training loop...')
                    ep = parameters['model_dict']['num_epochs']
                    met_tolerance = 1

        if ep > 0:
            if rank == 0 and parameters['io_dict']['training_info_nwrite_steps'] % ep == 0:
                print('Writing run information...')
                run_data = {
                    'epoch_timings': epoch_times,
                    'times_loader_shuffled': shuffle_counter,
                    'met_tolerance': met_tolerance,
                    'training_loss': L_train,
                    'validation_loss': L_valid,
                }
                save_dictionary(fname=os.path.join(cat.parameters['io_dict']['model_dir'], 'run_information.npy'),
                                data=run_data)
        ep += 1

    if parameters['device_dict']['run_ddp']:
        ddp_destroy()









