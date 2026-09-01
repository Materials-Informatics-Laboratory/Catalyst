"""
Smoke example 05: full Catalyst backend training -> checkpoint -> inference.

This is the end-to-end smoke example for Catalyst's public training backend. It
constructs a tiny ASE-based FCC Al dataset, serializes the graphs in Catalyst's
normal graph-file format, creates the train/validation/test sample files expected
by the backend, and then exercises::

    Catalyst(..., task=task)
        -> cat.set_model(...)
        -> catalyst.ml.training.run_training(...)
        -> checkpoint creation
        -> catalyst.ml.inference.run_inference(...)

The model is deliberately tiny so this remains deterministic and inexpensive on
CPU while still testing the actual Catalyst orchestration stack.

Run:
    python examples/gnn_examples/smoke/05_al_fcc_train_checkpoint_inference_smoke.py
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from catalyst.data.utils import load_dictionary, save_dictionary
from catalyst.ml.gnn import GNN, GNNTask
from catalyst.ml.inference import run_inference
from catalyst.ml.training import run_training
from catalyst.observer import Catalyst


class MeanBondLengthRegressor(nn.Module):
    """Tiny graph regressor using the mean Catalyst bond-distance feature."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, data):
        x_bnd = data.x_bnd.float().reshape(-1, 1)
        batch = getattr(data, "x_bnd_batch", None)
        if batch is None:
            batch = torch.zeros(
                x_bnd.size(0),
                dtype=torch.long,
                device=x_bnd.device,
            )

        # PyG's follow_batch machinery supplies one bond-batch label per edge.
        # Avoid max().item() here so the model remains friendly to compiled/GPU
        # execution. The number of graph IDs in the batch is already available.
        gid = getattr(data, "gid", None)
        if isinstance(gid, (list, tuple)):
            n_graphs = len(gid)
        elif torch.is_tensor(gid) and gid.ndim > 0:
            n_graphs = int(gid.shape[0])
        else:
            n_graphs = 1

        sums = x_bnd.new_zeros((n_graphs, 1))
        counts = x_bnd.new_zeros((n_graphs, 1))
        sums.index_add_(0, batch, x_bnd)
        counts.index_add_(0, batch, torch.ones_like(x_bnd))
        mean_bond = sums / counts.clamp_min(1.0)
        return self.linear(mean_bond).view(-1)


def make_graph(lattice_constant: float, index: int):
    """Build one small FCC Al Catalyst graph and deterministic scalar target."""
    from ase.build import bulk
    from catalyst.graph.alignnd import alignn_gen

    atoms = bulk("Al", "fcc", a=float(lattice_constant), cubic=True)
    atoms.pbc = True

    graph = alignn_gen(
        {
            "type": "alignnd",
            "raw_data": atoms,
            "element_list": ["Al"],
            "neighbor_params": [3.5, 12],
            "include_angs": False,
            "is_dihedral": False,
            "store_raw_data": False,
            "use_pt": False,
            "include_equivariant_fields": True,
            "auto_retry_graph": True,
            "max_graph_attempts": 4,
            "require_bonds": True,
            "require_angles": False,
            "require_dihedrals": False,
            "retry_verbose": False,
        }
    )

    mean_bond = float(graph.x_bnd.float().mean())

    # Exactly representable by MeanBondLengthRegressor. This makes loss
    # reduction deterministic without turning the smoke test into a long fit.
    target = 0.25 * mean_bond + 0.10

    graph.gid = f"al_fcc_backend_smoke_{index}"
    graph.target_scalar = torch.tensor([target], dtype=torch.float32)
    graph.y = graph.target_scalar.clone()
    return graph


def write_backend_dataset(root: Path):
    """Write graphs and Catalyst train/validation/test sample dictionaries."""
    data_dir = root / "graphs"
    samples_dir = root / "samples"
    results_dir = root / "results"

    data_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    graphs = [
        make_graph(a, i)
        for i, a in enumerate((3.90, 3.96, 4.02, 4.08, 4.14, 4.20))
    ]

    for graph in graphs:
        torch.save(graph, data_dir / f"{graph.gid}.pt")

    # Give both train and validation enough variation to demonstrate genuine
    # learning while keeping the dataset tiny.
    training_gids = [graph.gid for graph in graphs[:4]]
    validation_gids = [graph.gid for graph in graphs[4:]]
    test_gids = [graph.gid for graph in graphs]

    save_dictionary(
        samples_dir / "train_valid_split.npy",
        {
            "training": training_gids,
            "validation": validation_gids,
        },
    )
    save_dictionary(
        samples_dir / "test_data.npy",
        {
            "validation": test_gids,
        },
    )

    return data_dir, samples_dir, results_dir, len(test_gids)


def newest_checkpoint(model_dir: Path) -> Path:
    """Return the newest epoch checkpoint produced by the training backend."""
    checkpoints = sorted(
        model_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not checkpoints:
        raise AssertionError(
            f"Catalyst training produced no checkpoint in {model_dir}."
        )
    return checkpoints[-1]


def main() -> None:
    torch.manual_seed(11)

    task = GNNTask.graph_scalar(target_key="target_scalar")

    with TemporaryDirectory(prefix="catalyst_backend_smoke_") as tmpdir:
        root = Path(tmpdir)
        data_dir, samples_dir, results_dir, n_test = write_backend_dataset(root)

        # All runtime/training settings enter Catalyst through the constructor.
        # The GNNTask owns the prediction semantics and is validated separately.
        cat = Catalyst(
            parameters={
                "device_dict": {
                    "device": "cpu",
                    "use_amp": False,
                    "run_ddp": False,
                    "pin_memory": False,
                },
                "io_dict": {
                    "main_path": str(root),
                    "data_dir": str(data_dir),
                    "samples_dir": str(samples_dir),
                    "results_dir": str(results_dir),
                    "graph_read_format": 0,
                    "write_indv_pred": False,
                    "remove_old_model": False,
                    "training_info_nwrite_steps": 1,
                },
                "loader_dict": {
                    "shuffle_loader": False,
                    "batch_size": [4, 2],
                    "num_workers": 0,
                    "batch_mode": "graphs",
                },
                "model_dict": {
                    "num_epochs": 30,
                    "train_delta": 0.0,
                    "train_tolerance": 0.0,
                    "max_deltas": 0,
                    "worsen_tolerance": 10.0,
                    "strict_loss_policy": False,
                    "validation_interval": 1,
                    "compile_model": False,
                    "loss_params": {
                        "function": "MSELoss",
                    },
                    "optimizer_params": {
                        "optimizer": "Adam",
                        "implementation": "default",
                        "params_group": {
                            "lr": 0.03,
                        },
                    },
                },
            },
            task=task,
        )

        backend_model = GNN(
            MeanBondLengthRegressor(),
            torch.device("cpu"),
        )
        cat.set_model(backend_model)

        # ------------------------------------------------------------------
        # Public Catalyst training backend.
        # ------------------------------------------------------------------
        run_training(rank=0, cat=cat)

        model_dir = Path(cat.parameters["io_dict"]["model_dir"])
        checkpoint = newest_checkpoint(model_dir)
        run_info = load_dictionary(model_dir / "run_information.npy")

        validation_loss = list(run_info.get("validation_loss", []))
        assert len(validation_loss) >= 2, (
            "The backend smoke test expected at least two validation epochs, "
            f"but received {len(validation_loss)}."
        )
        assert min(validation_loss[1:]) < validation_loss[0], (
            "Catalyst backend training did not improve validation loss: "
            f"{validation_loss}."
        )

        # ------------------------------------------------------------------
        # Public Catalyst inference backend. setup_inference() reloads the
        # checkpoint and reads test_data.npy through the normal data pipeline.
        # ------------------------------------------------------------------
        inference = run_inference(
            model_name=str(checkpoint),
            rank=0,
            cat=cat,
            test=False,
        )

        predictions = inference["pred"]
        predicted_graphs = sum(
            len(batch) if isinstance(batch, list) else 1
            for batch in predictions
        )

        assert predicted_graphs == n_test, (
            f"Expected predictions for {n_test} test graphs, got {predicted_graphs}."
        )
        assert not inference["vec"]

        print(f"First validation loss: {validation_loss[0]:.6e}")
        print(f"Best validation loss:  {min(validation_loss):.6e}")
        print(f"Checkpoint:            {checkpoint.name}")
        print(f"Inference graphs:      {predicted_graphs}")
        print("05_al_fcc_train_checkpoint_inference_smoke passed.")


if __name__ == "__main__":
    main()
