import numpy as np
import torch

from ....graph.graph import Generic_Graph_Data, Atomic_Graph_Data
from ....ml.utils.pooling_algorithms import scatter_

class SODAS():
    def __init__(self, mod, ls_mod,pooling='mean'):
        super().__init__()

        self.model = mod
        self.model.eval()
        self.dim_model = ls_mod
        self.preprocess = None
        self.pooling = pooling

    def generate_gnn_latent_space(self, parameters, loader, global_data=True):
        print("Performing graph encodings...")

        device = parameters["device_dict"]["device"]

        follow_batch = getattr(loader, "follow_batch", [])
        total_batches = []

        with torch.inference_mode():
            for data in loader:
                data = data.to(device, non_blocking=True)
                preds = self.model(data)

                if isinstance(data, Atomic_Graph_Data):
                    if "x_ang" in follow_batch:
                        batch_vec = torch.cat(
                            (data.x_atm_batch, data.x_bnd_batch, data.x_ang_batch),
                            dim=0,
                        )
                    else:
                        batch_vec = torch.cat(
                            (data.x_atm_batch, data.x_bnd_batch),
                            dim=0,
                        )

                elif isinstance(data, Generic_Graph_Data):
                    if "edge_A" in follow_batch:
                        batch_vec = torch.cat(
                            (data.node_G_batch, data.node_A_batch, data.edge_A_batch),
                            dim=0,
                        )
                    else:
                        batch_vec = torch.cat(
                            (data.node_G_batch, data.node_A_batch),
                            dim=0,
                        )

                else:
                    raise TypeError(f"Unsupported graph data type: {type(data)}")

                if global_data:
                    rp = scatter_(preds, batch_vec, dim=0, reduce=self.pooling)
                else:
                    rp = scatter_(preds, batch_vec, dim=0)

                total_batches.append(rp.detach().cpu())

        return torch.cat(total_batches, dim=0).numpy()

    def fit_preprocess(self, data):
        print("Performing graph preprocessing...")
        from sklearn import preprocessing
        self.preprocess = preprocessing.StandardScaler().fit(data)

    def fit_dim_red(self, data, preprocess_data=True):
        print("Performing latent space conversion...")

        if preprocess_data:
            if self.preprocess is None:
                raise RuntimeError(
                    "preprocess_data=True, but fit_preprocess() has not been called."
                )
            data = self.preprocess.transform(data)

        self.dim_model.fit(data)

    def project_data(self, data, preprocess_data=True):
        print("Performing projections...")

        if preprocess_data:
            if self.preprocess is None:
                raise RuntimeError(
                    "preprocess_data=True, but fit_preprocess() has not been called."
                )
            data = self.preprocess.transform(data)

        return self.dim_model.transform(data)

















