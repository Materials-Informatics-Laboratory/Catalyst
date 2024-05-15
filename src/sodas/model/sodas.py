from torch_geometric.data import DataLoader
from ..nn.models.alignn import Encoder, Processor, Decoder, ALIGNN
import torch.nn as nn
import numpy as np

from torch_geometric.utils import scatter
from ..nn import MLP

class SODAS():
    def __init__(self, mod = ALIGNN(
                    encoder=Encoder(num_species=1, cutoff=3.0,act_func=nn.SiLU(), dim=100),
                    processor=Processor(num_convs=5, dim=100),
                    decoder=Decoder(node_dim=100, out_dim=10,act_func=nn.SiLU())
                 ls_mod=None):
        super().__init__()

        self.model = mod
        self.dim_model = ls_mod
        self.preprocess = None

    def generate_gnn_latent_space(self,loader,device=''):
        if device == '':
            print(f"Is CUDA supported by this system?{torch.cuda.is_available()}")
            print(f"CUDA version: {torch.version.cuda}")

            # Storing ID of current CUDA device
            cuda_id = torch.cuda.current_device()
            print(f"ID of current CUDA device:{torch.cuda.current_device()}")

            print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

            # Prepare model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = nn.DataParallel(self.model)
        self.model.eval()
        self.model.to(device)
        # Run a forward pass
        counter = 0
        total_data = []
        for data in loader:
            data = data.to(device)
            encoding = self.model.encoder(data)
            proc = self.model.processor(encoding)
            pred = self.model.decoder(proc)
            pred = scatter(pred,data.x_atm_batch, dim=0, reduce='mean')
            data.detach()
            tx = []
            for p in pred[0]:
                x = p.cpu().detach().numpy()
                tx.append(p.item())
            total_data.append(tx)
            counter += 1
        return np.array(total_data)

    def fit_preprocess(self,data):
        from sklearn import preprocessing
        self.preprocess = preprocessing.StandardScaler().fit(data)

    def fit_dim_red(self,data,preprocess_data=1):
        if preprocess_data:
            data = self.preprocess.transform(data)
        self.dim_model.fit(data)

    def project_data(self,data,preprocess_data=1):
        if preprocess_data:
            data = self.preprocess.transform(data)
        data = self.dim_model.transform(data)

        return data















