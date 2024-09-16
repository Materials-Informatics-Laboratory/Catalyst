from functools import partial
from torch import nn
import torch
import copy

from ..conv import MeshGraphNetsConv
from ..basis import gaussian, bessel
from ..conv import GatedGCN
from ..mlp import MLP

class Encoder(nn.Module):
    """ALIGNN/ALIGNN-d Encoder.
    The encoder must take a PyG graph object `data` and output the same `data`
    with additional fields `h_atm`, `h_bnd`, and `h_ang` that correspond to the atom, bond, and angle embedding.

    The input `data` must have three fields `x_atm`, `x_bnd`, and `x_ang` that describe the atom type
    (in onehot vectors), the bond lengths, and bond/dihedral angles (in radians).
    """
    def __init__(self, num_species, cutoff,act_func, dim=128, dihedral=False):
        super().__init__()
        self.num_species = num_species
        self.cutoff      = cutoff
        self.dim         = dim
        self.dihedral    = dihedral
        
        self.embed_atm = nn.Sequential(MLP([num_species, dim, dim], act=act_func), nn.LayerNorm(dim))
        self.embed_bnd = partial(bessel, start=0, end=cutoff, num_basis=dim)
        self.embed_ang = self.embed_ang_with_dihedral if dihedral else self.embed_ang_without_dihedral

    def embed_ang_with_dihedral(self, x_ang, mask_dih_ang):
        cos_ang = torch.cos(x_ang)
        sin_ang = torch.sin(x_ang)

        h_ang = torch.zeros([len(x_ang), self.dim], device=x_ang.device)
        h_ang[~mask_dih_ang, :self.dim//2] = gaussian(cos_ang[~mask_dih_ang], start=-1, end=1, num_basis=self.dim//2)

        h_cos_ang = gaussian(cos_ang[mask_dih_ang], start=-1, end=1, num_basis=self.dim//4)
        h_sin_ang = gaussian(sin_ang[mask_dih_ang], start=-1, end=1, num_basis=self.dim//4)
        h_ang[mask_dih_ang, self.dim//2:] = torch.cat([h_cos_ang, h_sin_ang], dim=-1)

        return h_ang
    
    def embed_ang_without_dihedral(self, x_ang, mask_dih_ang):
        cos_ang = torch.cos(x_ang)
        return gaussian(cos_ang, start=-1, end=1, num_basis=self.dim)

    def forward(self, data):
        # Embed atoms
        data.h_atm = self.embed_atm(data.x_atm)
        
        # Embed bonds
        data.h_bnd = self.embed_bnd(data.x_bnd)
        
        # Embed angles
        data.h_ang = self.embed_ang(data.x_ang, data.mask_dih_ang)
        
        return data


class Processor(nn.Module):
    """ALIGNN Processor.
    The processor updates atom, bond, and angle embeddings.
    """
    def __init__(self, num_convs, dim,conv_type='mesh'):
        super().__init__()
        self.num_convs = num_convs
        self.dim = dim

        if conv_type == 'mesh':
            self.atm_bnd_convs = nn.ModuleList([copy.deepcopy(MeshGraphNetsConv(dim, dim)) for _ in range(num_convs)])
            self.bnd_ang_convs = nn.ModuleList([copy.deepcopy(MeshGraphNetsConv(dim, dim)) for _ in range(num_convs)])
        elif conv_type == 'gcn':
            self.atm_bnd_convs = nn.ModuleList([copy.deepcopy(GatedGCN(dim, dim)) for _ in range(num_convs)])
            self.bnd_ang_convs = nn.ModuleList([copy.deepcopy(GatedGCN(dim, dim)) for _ in range(num_convs)])

    def forward(self, data):
        edge_index_G = data.edge_index_G
        edge_index_A = data.edge_index_A
        
        for i in range(self.num_convs):
            data.h_bnd, data.h_ang = self.bnd_ang_convs[i](data.h_bnd, edge_index_A, data.h_ang)
            data.h_atm, data.h_bnd = self.atm_bnd_convs[i](data.h_atm, edge_index_G, data.h_bnd)
    
        return data

class SODAS_Decoder(nn.Module):
    def __init__(self, node_dim, out_dim):
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.decoder = MLP([node_dim, node_dim, out_dim], act=nn.SiLU())

    def forward(self, data):
        return self.decoder(data.h_atm)

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim,act_func):
        super().__init__()
        self.node_dim = in_dim
        self.out_dim = out_dim
        self.decoder = MLP([in_dim, in_dim, out_dim], act=act_func)

    def forward(self, data):
        atm_scalars = self.transform_atm(data.h_atm)
        bnd_scalars = self.transform_bnd(data.h_bnd)
        ang_scalars = self.transform_bnd(data.h_ang)
        return [atm_scalars, bnd_scalars, ang_scalars]

class PositiveScalarsDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.transform_atm = nn.Sequential(MLP([dim, dim, 1], act=nn.SiLU()), nn.Softplus())
        self.transform_bnd = nn.Sequential(MLP([dim, dim, 1], act=nn.SiLU()), nn.Softplus())
        self.transform_ang = nn.Sequential(MLP([dim, dim, 1], act=nn.SiLU()), nn.Softplus())

    def forward(self, data):
        atm_scalars = self.transform_atm(data.h_atm)
        bnd_scalars = self.transform_bnd(data.h_bnd)
        ang_scalars = self.transform_bnd(data.h_ang)
        return [atm_scalars, bnd_scalars, ang_scalars]


class ALIGNN(nn.Module):
    """ALIGNN model.
    Can optinally encode dihedral angles.
    """
    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder
    
    def forward(self, data):
        data = self.encoder(data)
        data = self.processor(data)
        return self.decoder(data)
