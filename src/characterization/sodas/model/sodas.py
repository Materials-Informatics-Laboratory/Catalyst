import torch.nn as nn
import numpy as np
import torch

from torch_geometric.utils import scatter

from ....graph.graph import Generic_Graph_Data, Atomic_Graph_Data
from ....ml.utils.predict import accumulate_predictions
class SODAS():
    def __init__(self, mod, ls_mod):
        super().__init__()

        self.model = mod
        self.model.eval()
        self.dim_model = ls_mod
        self.preprocess = None
        self.device = 'cpu'

    def send_model(self,device='cuda'):
        self.device = device
        self.model.to(device)
        if device == 'cpu':
            torch.cuda.empty_cache()

    def generate_gnn_latent_space(self,parameters,loader,global_data=True):
        # Run a forward pass
        print('Performing graph encodings...')
        total_data = []
        for data in loader:
            data.to(parameters['device_dict']['device'], non_blocking=True)
            preds = self.model(data)
                #for i,pred in enumerate(preds):
            if global_data:
                if isinstance(data,Atomic_Graph_Data):
                    if  'x_ang' in loader.follow_batch:
                        rp = scatter(preds,torch.cat((data.x_atm_batch,data.x_bnd_batch,data.x_ang_batch),0), dim=0, reduce='mean')
                    else:
                        rp = scatter(preds, torch.cat((data.x_atm_batch, data.x_bnd_batch), 0), dim=0,
                                               reduce='mean')
                elif isinstance(data,Generic_Graph_Data):
                    if 'edge_A' in loader.follow_batch:
                        rp = scatter(preds,torch.cat((data.node_G_batch,data.node_A_batch,data.edge_A_batch),0), dim=0, reduce='mean')
                    else:
                        rp = scatter(preds, torch.cat((data.node_G_batch, data.node_A_batch), 0), dim=0,
                                               reduce='mean')
            else:
                if isinstance(data,Atomic_Graph_Data):
                    if  'x_ang' in loader.follow_batch:
                        rp = scatter(preds,torch.cat((data.x_atm_batch,data.x_bnd_batch,data.x_ang_batch),0), dim=0)
                    else:
                        rp = scatter(preds, torch.cat((data.x_atm_batch, data.x_bnd_batch), 0), dim=0)
                elif isinstance(data,Generic_Graph_Data):
                    if 'edge_A' in loader.follow_batch:
                        rp = scatter(preds,torch.cat((data.node_G_batch,data.node_A_batch,data.edge_A_batch),0), dim=0)
                    else:
                        rp = scatter(preds, torch.cat((data.node_G_batch, data.node_A_batch), 0), dim=0)
            for tensor in rp:
                total_data.append(tensor.cpu().tolist())
        return np.array(total_data)

    def fit_preprocess(self,data):
        print('Performing graph preprocessing...')
        from sklearn import preprocessing
        self.preprocess = preprocessing.StandardScaler().fit(data)

    def fit_dim_red(self,data,preprocess_data=1):
        print('Performing latent space conversion...')
        if preprocess_data:
            data = self.preprocess.transform(data)
        self.dim_model.fit(data)

    def project_data(self,data,preprocess_data=1):
        print('Performing projections...')
        if preprocess_data:
            data = self.preprocess.transform(data)
        data = self.dim_model.transform(data)

        return data

    def clear_model(self):
        self.model.cpu()
        del self.model
        if self.device == 'cuda':
            torch.cuda.empty_cache()
















