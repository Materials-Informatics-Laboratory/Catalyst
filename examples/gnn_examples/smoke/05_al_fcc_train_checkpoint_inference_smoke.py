"""
Smoke example 05: graph construction -> Catalyst training -> checkpoint -> inference.

This intentionally uses a tiny graph-level regressor so the smoke test remains
fast and deterministic on CPU while exercising the high-level Catalyst GNN
training/checkpoint/inference backend on real ASE-built atomic graphs.

Run:
    python examples/gnn_examples/smoke/05_al_fcc_train_checkpoint_inference_smoke.py
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from catalyst.ml.gnn import GNN, GNNTask


class MeanBondLengthRegressor(nn.Module):
    """One-layer graph regressor using the mean Catalyst bond-distance feature."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, data):
        x_bnd = data.x_bnd.float().reshape(-1, 1)
        batch = getattr(data, "x_bnd_batch", None)
        if batch is None:
            batch = torch.zeros(x_bnd.size(0), dtype=torch.long, device=x_bnd.device)

        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        sums = x_bnd.new_zeros((n_graphs, 1))
        counts = x_bnd.new_zeros((n_graphs, 1))
        sums.index_add_(0, batch, x_bnd)
        counts.index_add_(0, batch, torch.ones_like(x_bnd))
        mean_bond = sums / counts.clamp_min(1.0)
        return self.linear(mean_bond).view(-1)


def make_graph(lattice_constant: float, index: int):
    from ase.build import bulk
    from catalyst.graph.alignnd import alignn_gen

    atoms = bulk("Al", "fcc", a=float(lattice_constant), cubic=True)
    atoms.pbc = True

    graph = alignn_gen(
        {
            "type": "alignnd",
            "raw_data": atoms,
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
    # A deterministic linear target that the one-layer smoke model can learn.
    target = 0.25 * mean_bond + 0.10

    graph.gid = f"al_fcc_train_smoke_{index}"
    graph.target_scalar = torch.tensor([target], dtype=torch.float32)
    graph.y = graph.target_scalar.clone()
    return graph


def make_parameters(task: GNNTask, model_dir: Path):
    parameters = {
        "device_dict": {
            "device": "cpu",
            "use_amp": False,
            "run_ddp": False,
            "pin_memory": False,
        },
        "io_dict": {
            "model_dir": str(model_dir),
            "results_dir": str(model_dir),
            "write_indv_pred": False,
            "remove_old_model": False,
        },
        "model_dict": {
            "accumulate_loss": "exact",
            "prediction_params": {},
            "loss_params": {"function": nn.MSELoss()},
            "optimizer_params": {
                "optimizer": "Adam",
                "params_group": {"lr": 0.05},
            },
        },
    }
    task.apply_to_catalyst_parameters(parameters)
    return parameters


def main() -> None:
    from torch_geometric.loader import DataLoader

    torch.manual_seed(11)

    graphs = [
        make_graph(a, i)
        for i, a in enumerate((3.90, 3.96, 4.02, 4.08, 4.14, 4.20))
    ]
    loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False, follow_batch=["x_bnd"])

    task = GNNTask.graph_scalar(target_key="target_scalar")

    with TemporaryDirectory(prefix="catalyst_smoke_") as tmpdir:
        model_dir = Path(tmpdir)
        parameters = make_parameters(task, model_dir)

        model = MeanBondLengthRegressor()
        trainer = GNN(model, torch.device("cpu"))
        trainer.training_loader = loader
        trainer.validation_loader = loader
        trainer.set_optimizer_(parameters)

        initial_loss = trainer.validate(parameters)
        final_loss = initial_loss

        for _ in range(40):
            trainer.train({"params": parameters})
            final_loss = trainer.validate(parameters)
            if final_loss < initial_loss * 0.05:
                break

        assert final_loss < initial_loss, (
            f"Training smoke test did not improve: initial={initial_loss}, final={final_loss}."
        )

        # set_optimizer_ inserts a live parameter iterator into the config. Remove
        # it before checkpoint serialization so save_checkpoint can deepcopy the
        # portable configuration dictionary.
        parameters["model_dict"]["optimizer_params"]["params_group"].pop("params", None)

        checkpoint = model_dir / "smoke_checkpoint.pt"
        trainer.save_checkpoint(parameters, epoch=0, fname=checkpoint)
        assert checkpoint.is_file()

        batch = next(iter(loader))
        with torch.no_grad():
            prediction_before_reload = trainer.model(batch).detach().clone()

        reloaded = GNN(MeanBondLengthRegressor(), torch.device("cpu"))
        reloaded.load_checkpoint(
            fname=checkpoint,
            map_location="cpu",
            load_optimizer=False,
            strict=True,
        )
        reloaded.validation_loader = loader

        with torch.no_grad():
            prediction_after_reload = reloaded.model(batch).detach().clone()

        torch.testing.assert_close(
            prediction_after_reload,
            prediction_before_reload,
            rtol=1.0e-6,
            atol=1.0e-7,
        )

        inference = reloaded.predict(parameters)
        assert len(inference["pred"]) == len(loader)
        assert not inference["vec"]

        print(f"Initial validation loss: {initial_loss:.6e}")
        print(f"Final validation loss:   {final_loss:.6e}")
        print("05_al_fcc_train_checkpoint_inference_smoke passed.")


if __name__ == "__main__":
    main()
