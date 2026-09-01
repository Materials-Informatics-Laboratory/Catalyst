"""High-level Catalyst training orchestration.

The supported JOSS-facing workflow is ordinary training plus checkpoint resume.
The pre-2.x active-learning routine depended on removed legacy APIs and is kept
only as an explicit compatibility stub rather than silently failing at runtime.
"""

from __future__ import annotations

import copy
import glob
import os
import re
import shutil
import sys
import time

import torch

from ..data.utils import load_dictionary, save_dictionary
from .utils.distributed import (
    ddp_destroy,
    ddp_model,
    ddp_setup,
    validate_ddp_configuration,
)
from .utils.memory import optimizer_to


def _training_model_dir(parameters):
    return os.path.join(parameters["io_dict"]["main_path"], "models", "training")


def _checkpoint_epoch(path):
    match = re.search(r"checkpoint_epoch_(\d+)\.pt$", os.path.basename(os.fspath(path)))
    return int(match.group(1)) if match else -1


def _resolve_restart_checkpoint(parameters, model_dir):
    requested = parameters["io_dict"].get("loaded_model_name")
    if requested:
        requested = os.fspath(requested)
        candidates = [requested]
        if not os.path.isabs(requested):
            candidates.extend(
                [
                    os.path.join(model_dir, requested),
                    os.path.join(parameters["io_dict"]["main_path"], requested),
                ]
            )
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        raise FileNotFoundError(
            "restart_training=True but loaded_model_name could not be found: "
            f"{requested!r}."
        )

    candidates = glob.glob(os.path.join(model_dir, "checkpoint_epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            "restart_training=True but no checkpoint_epoch_*.pt file exists in "
            f"{model_dir!r}. Set io_dict['loaded_model_name'] explicitly."
        )
    return max(candidates, key=lambda path: (_checkpoint_epoch(path), os.path.getmtime(path)))


def _load_existing_run_history(model_dir):
    fname = os.path.join(model_dir, "run_information.npy")
    if not os.path.isfile(fname):
        return [], [], [], [], 0, []
    try:
        info = load_dictionary(fname)
        return (
            list(info.get("training_loss", [])),
            list(info.get("validation_loss", [])),
            list(info.get("epoch_timings", [])),
            list(info.get("validation_deltas", [])),
            int(info.get("met_tolerance", 0)),
            list(info.get("validation_epochs", [])),
        )
    except (OSError, ValueError, TypeError, KeyError):
        # A stale/old run-information file should not prevent checkpoint resume.
        return [], [], [], [], 0, []


def _write_run_information(cat, epoch_times, running_valid_delta, met_tolerance,
                           training_loss, validation_loss, validation_epochs=None):
    save_dictionary(
        fname=os.path.join(cat.parameters["io_dict"]["model_dir"], "run_information.npy"),
        data={
            "epoch_timings": epoch_times,
            "validation_deltas": running_valid_delta,
            "met_tolerance": met_tolerance,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_epochs": list(validation_epochs or []),
        },
    )


def setup_training(rank, cat=None):
    """Prepare data/DDP and the training output directory.

    Fresh training clears the old training directory on rank 0.  Restart mode
    intentionally preserves it so the checkpoint being resumed cannot be
    deleted before it is loaded.
    """
    if cat is None:
        raise ValueError("setup_training requires a Catalyst parameter object via cat=...")
    parameters = cat.parameters
    restart = bool(parameters["model_dict"].get("restart_training", False))
    model_dir = _training_model_dir(parameters)
    parameters["io_dict"]["model_dir"] = model_dir

    model = parameters["model_dict"].get("model")
    if model is None:
        raise ValueError("parameters['model_dict']['model'] must be set before training.")

    # Mandatory final staged-validation boundary.  This resolves/binds the task
    # if possible and validates configuration + task + model + runtime before
    # deleting/creating output directories, graph loading, optimizer allocation,
    # or epoch 1 can begin.
    if hasattr(cat, "validate_ready_for_training"):
        cat.validate_ready_for_training(model=model, rank=rank)
        parameters = cat.parameters

    validate_ddp_configuration(parameters, rank=rank)

    if rank == 0:
        print("Restarting model training..." if restart else "Training model...")
        if os.path.isdir(model_dir) and not restart:
            shutil.rmtree(model_dir)
        os.makedirs(model_dir, exist_ok=True)

    model.load_data(
        parameters,
        samples_file=os.path.join(parameters["io_dict"]["samples_dir"], "train_valid_split.npy"),
        format=parameters["io_dict"]["graph_read_format"],
        rank=rank,
    )

    if parameters["device_dict"]["run_ddp"]:
        ddp_setup(
            rank,
            parameters["device_dict"]["world_size"],
            parameters["device_dict"]["ddp_backend"],
        )

    return model_dir


def run_active_learning(rank, cat=None):
    """Compatibility stub for the removed legacy active-learning workflow.

    The historical implementation depended on APIs removed during the 2.x GNN
    refactor (``setup_model``, ``test_non_intepretable_internal``, legacy model
    serialization, etc.).  Failing explicitly prevents users from obtaining a
    partially executed or scientifically ambiguous active-learning run.
    """
    del rank, cat
    raise NotImplementedError(
        "Catalyst's legacy active-learning driver is not part of the supported "
        "2.2/JOSS API because it depended on removed pre-2.x training internals. "
        "Use run_training() for training/restart workflows. A redesigned active-"
        "learning API should be introduced as a separately tested feature."
    )


def run_training(rank, cat=None):
    """Train a Catalyst GNN, optionally resuming a saved checkpoint."""
    if cat is None:
        raise ValueError("run_training requires a Catalyst parameter object via cat=...")

    parameters = cat.parameters
    restart = bool(parameters["model_dict"].get("restart_training", False))
    model_dir = setup_training(rank=rank, cat=cat)

    model = parameters["model_dict"]["model"]
    model.device = parameters["device_dict"]["device"]
    model.model.to(model.device)
    if hasattr(model, "configure_numeric_performance"):
        model.configure_numeric_performance(parameters)

    compile_enabled = bool(parameters["model_dict"].get("compile_model", False))
    if compile_enabled:
        model.compile_model(
            backend=parameters["model_dict"].get("compile_backend", "inductor"),
            mode=parameters["model_dict"].get("compile_mode", "default"),
            dynamic=parameters["model_dict"].get("compile_dynamic", True),
            suppress_errors=parameters["model_dict"].get("compile_suppress_errors", False),
        )

    if parameters["device_dict"]["run_ddp"]:
        model.model = ddp_model(
            model=model.model,
            find_unused_parameters=parameters["device_dict"]["find_unused_parameters"],
            rank=rank,
            batchnorm=parameters["model_dict"].get("batchnorm", False),
            gradient_as_bucket_view=parameters["device_dict"].get(
                "ddp_gradient_as_bucket_view", False
            ),
            static_graph=parameters["device_dict"].get("ddp_static_graph", False),
            bucket_cap_mb=parameters["device_dict"].get("ddp_bucket_cap_mb", None),
        )

    model.set_optimizer_(parameters=parameters)

    ep = 0
    L_train, L_valid = [], []
    epoch_times, running_valid_delta = [], []
    validation_epochs = []
    met_tolerance = 0

    if restart:
        checkpoint_name = _resolve_restart_checkpoint(parameters, model_dir)
        loaded_epoch = model.load_checkpoint(
            checkpoint_name,
            map_location=model.device,
            load_optimizer=True,
            strict=True,
        )
        ep = 0 if loaded_epoch is None else int(loaded_epoch) + 1
        parameters["io_dict"]["loaded_model_name"] = checkpoint_name
        (
            L_train,
            L_valid,
            epoch_times,
            running_valid_delta,
            met_tolerance,
            validation_epochs,
        ) = _load_existing_run_history(model_dir)
        if rank == 0:
            print(f"Resumed checkpoint {checkpoint_name} at epoch {ep}.")

    # Every rank loads the same shared run history so restart decisions and
    # historical minima remain identical across DDP ranks.
    min_loss_train = min(L_train) if L_train else float("inf")
    min_loss_valid = min(L_valid) if L_valid else float("inf")

    model.set_dataloader(cat=cat, epoch=ep)

    if restart:
        core = model._core_model() if hasattr(model, "_core_model") else model.model
        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in core.state_dict().items()
        }
        best_optimizer_state = copy.deepcopy(model.optimizer.state_dict())
    else:
        best_model_state = None
        best_optimizer_state = None
    patience_counter = 0
    patience = int(parameters["model_dict"].get("patience", 0))
    worsen_tolerance = float(parameters["model_dict"].get("worsen_tolerance", 0.0))
    if worsen_tolerance < 0:
        raise ValueError("model_dict['worsen_tolerance'] must be >= 0.")

    num_epochs = int(parameters["model_dict"]["num_epochs"])
    validation_interval = int(parameters["model_dict"].get("validation_interval", 1))
    if validation_interval < 1:
        raise ValueError("model_dict['validation_interval'] must be >= 1.")
    max_deltas = int(parameters["model_dict"]["max_deltas"])
    strict_loss_policy = bool(parameters["model_dict"].get("strict_loss_policy", False))

    while ep < num_epochs:
        start_time = time.time()
        if rank == 0:
            print(f"Epoch {ep + 1} of {num_epochs}")
            sys.stdout.flush()

        if parameters["device_dict"]["run_ddp"]:
            if hasattr(model.training_loader, "sampler") and hasattr(model.training_loader.sampler, "set_epoch"):
                model.training_loader.sampler.set_epoch(ep)
            if hasattr(model.validation_loader, "sampler") and hasattr(model.validation_loader.sampler, "set_epoch"):
                model.validation_loader.sampler.set_epoch(ep)

        loss_train = float(model.train(training_dict={"params": parameters}))
        L_train.append(loss_train)

        should_validate = (
            validation_interval == 1
            or (ep + 1) % validation_interval == 0
            or ep == num_epochs - 1
        )
        loss_valid = None
        if should_validate:
            loss_valid = float(model.validate(parameters=parameters, rank=rank))
            L_valid.append(loss_valid)
            validation_epochs.append(ep)

            validation_improved = loss_valid < min_loss_valid
            training_improved = loss_train < min_loss_train
            accepted_improvement = validation_improved and (
                training_improved if strict_loss_policy else True
            )

            if accepted_improvement:
                min_loss_valid = loss_valid
                min_loss_train = loss_train if strict_loss_policy else min(min_loss_train, loss_train)
                core = model._core_model() if hasattr(model, "_core_model") else model.model
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in core.state_dict().items()
                }
                best_optimizer_state = copy.deepcopy(model.optimizer.state_dict())
                if rank == 0:
                    print("Saving model checkpoint...")
                    model.save_checkpoint(parameters, ep, rank=rank)
                patience_counter = 0
            else:
                patience_counter += 1
                if (
                    best_model_state is not None
                    and np_is_finite(min_loss_valid)
                    and loss_valid > min_loss_valid * (1.0 + worsen_tolerance)
                ):
                    if rank == 0:
                        print(
                            "Validation worsened by more than "
                            f"{100.0 * worsen_tolerance:.1f}%. Reverting to best model."
                        )
                    core = model._core_model() if hasattr(model, "_core_model") else model.model
                    core.load_state_dict(best_model_state)
                    if best_optimizer_state is not None:
                        model.optimizer.load_state_dict(best_optimizer_state)
                        optimizer_to(model.optimizer, model.device)
                    patience_counter = 0

            if len(L_valid) > 1:
                running_valid_delta.append(abs(L_valid[-1] - L_valid[-2]))
                if len(running_valid_delta) > max_deltas:
                    running_valid_delta.pop(0)

                if max_deltas > 0 and len(running_valid_delta) == max_deltas:
                    avg_delta = sum(running_valid_delta) / max_deltas
                    avg_valid = sum(L_valid[-max_deltas:]) / max_deltas
                    if rank == 0:
                        print(f"Running validation delta = {avg_delta}")
                    if (
                        avg_delta < parameters["model_dict"]["train_delta"]
                        and avg_valid < parameters["model_dict"]["train_tolerance"]
                    ):
                        if rank == 0:
                            print("Validation delta satisfies set tolerance... exiting training loop...")
                        met_tolerance = 1
                        break

        if rank == 0:
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
            valid_text = "not run" if loss_valid is None else str(loss_valid)
            print(
                f"Train loss = {loss_train} Validation loss = {valid_text} "
                f"epoch_time = {epoch_time:.3f} seconds"
            )

        write_steps = int(parameters["io_dict"].get("training_info_nwrite_steps", 0) or 0)
        if rank == 0 and write_steps > 0 and (ep + 1) % write_steps == 0:
            print("Writing run information...")
            _write_run_information(
                cat, epoch_times, running_valid_delta, met_tolerance, L_train, L_valid,
                validation_epochs=validation_epochs,
            )

        # Preserve the existing patience parameter as a monitoring signal.  The
        # GNN optimizer currently has no scheduler object wired to this driver,
        # so stopping/reducing LR implicitly would be a behavior change.
        if patience > 0 and patience_counter >= patience and rank == 0:
            print(f"No accepted improvement for {patience_counter} epochs.")

        ep += 1

    if rank == 0:
        _write_run_information(
            cat, epoch_times, running_valid_delta, met_tolerance, L_train, L_valid,
            validation_epochs=validation_epochs,
        )

    if parameters["device_dict"]["run_ddp"]:
        ddp_destroy()


def np_is_finite(value):
    # Tiny helper avoids introducing NumPy into the training control path.
    return value != float("inf") and value != float("-inf") and value == value
