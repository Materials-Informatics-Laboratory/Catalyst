import torch.nn as nn
import torch
import math

def loss_setup(params):
    if isinstance(params['function'],str):
        if params['function'] == 'MaxNpercent':
            return MaxNpercent(percent=params['percent'],sub_function=params['sub_function'])
    else:
        return params['function']

class MaxNpercent(nn.Module):
    def __init__(self, percent, sub_function):
        super(MaxNpercent, self).__init__()
        self.percent = percent
        self.sub_function = sub_function
    def forward(self, input, target):
        # Compute the loss
        n = math.ceil(self.percent*float(len(input)))
        stacked_tensor = torch.stack([input,target])
        diff_tensor = torch.diff(stacked_tensor, dim=0)
        sorted_indices = torch.argsort(diff_tensor,descending=True)[:n]

        sorted_inputs = input[sorted_indices]
        sorted_targets = target[sorted_indices]

        loss = self.sub_function(sorted_inputs,sorted_targets)

        return loss















