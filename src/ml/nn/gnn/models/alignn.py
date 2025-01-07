from functools import partial
from torch import nn
import torch
import copy

from ..conv import MeshGraphNetsConv
from src.ml.nn.utils.basis import gaussian, bessel, scalar2basis
from ..conv import GatedGCN
from src.ml.nn.mlp import MLP

from src.graph.graph import Generic_Graph_Data, Atomic_Graph_Data

class Encoder_atomic(nn.Module):
    def __init__(self, num_species, cutoff,act_func, dim=128, dihedral=False,params_group=None):
        super().__init__()
        self.num_species = num_species
        self.cutoff      = cutoff
        self.dim         = dim
        self.dihedral    = dihedral
        
        self.embed_g_node = nn.Sequential(MLP([num_species, dim, dim], act=act_func), nn.LayerNorm(dim))
        self.embed_a_node = partial(bessel, start=0, end=cutoff, num_basis=dim)
        self.embed_a_edge = self.embed_ang_with_dihedral if dihedral else self.embed_ang_without_dihedral

    def embed_ang_with_dihedral(self, x_ang, mask_dih_ang):
        cos_ang = torch.cos(x_ang)
        sin_ang = torch.sin(x_ang)

        h_ang = torch.zeros([len(x_ang), self.dim], device=x_ang.device)
        h_ang[~mask_dih_ang, :self.dim//2] = gaussian(cos_ang[~mask_dih_ang], start=-1, end=1, num_basis=self.dim//2)

        h_cos_ang = gaussian(cos_ang[mask_dih_ang], start=-1, end=1, num_basis=self.dim//4)
        h_sin_ang = gaussian(sin_ang[mask_dih_ang], start=-1, end=1, num_basis=self.dim//4)
        h_ang[mask_dih_ang, self.dim//2:] = torch.cat([h_cos_ang, h_sin_ang], dim=-1)

        return h_ang
    
    def embed_ang_without_dihedral(self, x_ang):
        cos_ang = torch.cos(x_ang)
        return gaussian(cos_ang, start=-1, end=1, num_basis=self.dim)

    def forward(self, data):
        if isinstance(data,Atomic_Graph_Data):
            # Embed atoms
            data.h_atm = self.embed_g_node(data.x_atm)

            # Embed bonds
            data.h_bnd = self.embed_a_node(data.x_bnd)

            # Embed angles
            if hasattr(data,'x_ang'):
                data.h_ang = self.embed_a_edge(data.x_ang)
        elif isinstance(data,Generic_Graph_Data):
            data.h_g_node = self.embed_g_node(data.node_G)
            data.h_a_node = self.embed_a_node(data.node_A)
            if hasattr(data,'edge_A'):
                data.h_a_edge = self.embed_a_edge((data.edge_A))
        return data

class Encoder_generic(nn.Module):
    def __init__(self, act_func, dim=128,basis='gaussian', params_group=None):
        super().__init__()
        self.dim = dim
        self.a_node_len = 0
        self.a_edge_len = 0
        self.g_node_len = 0
        self.act_func = act_func
        self.embed_g_node = None
        self.embed_a_node = None
        self.embed_a_edge = None
        self.basis = basis
        self.initialize_mlp = True

    def initialize_ml(self,g_len,device='cpu',a_edge=False):
        if self.initialize_mlp:
            self.g_node_len = g_len
            self.embed_g_node = nn.Sequential(MLP([self.g_node_len, self.dim, self.dim], act=self.act_func), nn.LayerNorm(self.dim)).to(device)
            self.embed_a_node = nn.Sequential(MLP([self.dim, self.dim, self.dim], act=self.act_func), nn.LayerNorm(self.dim)).to(device)
            if a_edge:
                self.embed_a_edge = nn.Sequential(MLP([self.dim, self.dim, self.dim], act=self.act_func), nn.LayerNorm(self.dim)).to(device)
            self.initialize_mlp = False

    def forward(self, data):
        if isinstance(data, Atomic_Graph_Data):
            self.initialize_ml(len(data.x_atm[0]),a_edge=hasattr(data, 'x_ang'),device=data.x_atm[0].device)
            data.h_atm = self.embed_g_node(data.x_atm)
            data.h_bnd = self.embed_a_node(data.x_bnd)
            if hasattr(data, 'x_ang'):
                data.h_ang = self.embed_a_edge(data.x_ang)
        elif isinstance(data, Generic_Graph_Data):
            self.initialize_ml(len(data.node_G[0]), a_edge=hasattr(data, 'edge_A'),device=data.node_G[0].device)
            data.h_g_node = self.embed_g_node(data.node_G)
            a_node_x = torch.tensor(scalar2basis(data.node_A,start=-1,end=1,num_basis=self.dim,basis=self.basis),device=data.node_A.device)
            data.h_a_node = self.embed_a_node(a_node_x)
            if hasattr(data, 'edge_A'):
                a_edge_x = torch.tensor(scalar2basis(data.edge_A, start=-1, end=1, num_basis=self.dim, basis=self.basis),device=data.edge_A.device)
                data.h_a_edge = self.embed_a_edge(a_edge_x)
        return data

class Processor(nn.Module):
    def __init__(self, num_convs, dim,conv_type='mesh',aggr='add',encode_a=1,act=None):
        super().__init__()
        self.num_convs = num_convs
        self.dim = dim
        self.aggr = aggr
        self.act = act
        self.conv = conv_type
        self.ang = encode_a

        if self.conv == 'mesh':
            self.g_convs = nn.ModuleList([copy.deepcopy(MeshGraphNetsConv(self.dim, self.dim,aggr_scheme=self.aggr)) for _ in range(self.num_convs)])
            if self.ang:
                self.a_convs = nn.ModuleList([copy.deepcopy(MeshGraphNetsConv(self.dim, self.dim,aggr_scheme=self.aggr)) for _ in range(self.num_convs)])
        elif self.conv == 'gcn':
            self.g_convs = nn.ModuleList([copy.deepcopy(GatedGCN(self.dim, self.dim,aggr_scheme=self.aggr,act=self.act)) for _ in range(self.num_convs)])
            if self.ang:
                self.a_convs = nn.ModuleList([copy.deepcopy(GatedGCN(self.dim, self.dim,aggr_scheme=self.aggr,act=self.act)) for _ in range(self.num_convs)])

    def forward(self, data):
        if isinstance(data,Atomic_Graph_Data):
            edge_index_G = data.edge_index_G
            if hasattr(data, 'x_ang'):
                edge_index_A = data.edge_index_A
            for i in range(self.num_convs):
                if hasattr(data, 'x_ang'):
                    data.h_bnd, data.h_ang = self.a_convs[i](data.h_bnd, edge_index_A, data.h_ang)
                data.h_atm, data.h_bnd = self.g_convs[i](data.h_atm, edge_index_G, data.h_bnd)
        elif isinstance(data,Generic_Graph_Data):
            edge_index_G = data.edge_index_G
            if hasattr(data, 'edge_A'):
                edge_index_A = data.edge_index_A
            for i in range(self.num_convs):
                if hasattr(data, 'edge_A'):
                    data.h_a_node, data.h_a_edge = self.a_convs[i](data.h_a_node, edge_index_A, data.h_a_edge)
                data.h_g_node, data.h_a_node = self.g_convs[i](data.h_g_node, edge_index_G, data.h_a_node)
        return data

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim,act_func,combine=True):
        super().__init__()
        self.node_dim = in_dim
        self.out_dim = out_dim
        self.combine = combine
        self.decoder = MLP([in_dim, in_dim, out_dim], act=act_func)

    def forward(self, data):
        if isinstance(data,Atomic_Graph_Data):
            atm_scalars = self.decoder(data.h_atm)
            bnd_scalars = self.decoder(data.h_bnd)
            if hasattr(data, 'x_ang'):
                ang_scalars = self.decoder(data.h_ang)
                if self.combine:
                    return torch.cat((atm_scalars,bnd_scalars,ang_scalars),0)
                else:
                    return [atm_scalars, bnd_scalars, ang_scalars]
            elif hasattr(data, 'x_bnd'):
                if self.combine:
                    return torch.cat((atm_scalars,bnd_scalars),0)
                else:
                    return [atm_scalars, bnd_scalars]
        elif isinstance(data,Generic_Graph_Data):
            g_node_scalars = self.decoder(data.h_g_node)
            a_node_scalars = self.decoder(data.h_a_node)
            if hasattr(data, 'edge_A'):
                a_edge_scalars = self.decoder(data.h_a_edge)
                if self.combine:
                    return torch.cat((g_node_scalars, a_node_scalars, a_edge_scalars), 0)
                else:
                    return [g_node_scalars, a_node_scalars, a_edge_scalars]
            elif hasattr(data, 'node_A'):
                if self.combine:
                    return torch.cat((g_node_scalars, a_node_scalars), 0)
                else:
                    return [g_node_scalars, a_node_scalars]

class PositiveScalarsDecoder(nn.Module):
    def __init__(self, dim,act_func):
        super().__init__()
        self.dim = dim
        self.transform_g_node = nn.Sequential(MLP([dim, dim, 1], act=act_func), nn.Softplus())
        self.transform_a_node = nn.Sequential(MLP([dim, dim, 1], act=act_func), nn.Softplus())
        self.transform_a_edge = nn.Sequential(MLP([dim, dim, 1], act=act_func), nn.Softplus())

    def forward(self, data):
        if isinstance(data,Atomic_Graph_Data):
            atm_scalars = self.transform_g_node(data.h_atm)
            bnd_scalars = self.transform_a_node(data.h_bnd)
            if hasattr(data, 'x_ang'):
                ang_scalars = self.transform_a_edge(data.h_ang)
                return [atm_scalars, bnd_scalars, ang_scalars]
            else:
                return [atm_scalars, bnd_scalars]
        elif isinstance(data,Generic_Graph_Data):
            g_node_scalars = self.transform_g_node(data.h_g_node)
            a_node_scalars = self.transform_a_node(data.h_a_node)
            if hasattr(data, 'edge_A'):
                a_edge_scalars = self.transform_a_edge(data.h_a_edge)
                return [g_node_scalars, a_node_scalars, a_edge_scalars]
            else:
                return [g_node_scalars, a_node_scalars]

class ALIGNN(nn.Module):

    def __init__(self, encoder, processor, decoder):
        super().__init__()
        self.encoder   = encoder
        self.processor = processor
        self.decoder   = decoder
    
    def forward(self, data):
        data = self.encoder(data)
        data = self.processor(data)
        return self.decoder(data)
