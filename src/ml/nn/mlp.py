import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, hs, act=None):
        super().__init__()
        self.hs = hs
        self.act = act
        
        num_layers = len(hs)

        layers = []
        for i in range(num_layers-1):
            layers += [nn.Linear(hs[i], hs[i+1])]
            if (act is not None) and (i < num_layers-2):
                layers += [act]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(hs={self.hs}, act={self.act})'