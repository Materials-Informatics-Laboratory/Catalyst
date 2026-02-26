import torch.nn as nn
import torch
import math
from ..gnn.modules.utils.predict import accumulate_predictions
def loss_setup(params):
    if isinstance(params['function'],str):
        if params['function'] == 'MaxNpercent':
            return MaxNpercent(percent=params['percent'],sub_function=params['sub_function'])
    else:
        return params['function']

def active_learning_setup(params,model_dict):
    if 'loss_regularization' in params['model_dict']['active_learning_params_group']['training_params_group']:
        if params['model_dict']['active_learning_params_group']['training_params_group']['loss_regularization'] == 'EWC':
            return {
                    'loss_regularization': EWC(model_dict['model'], model_dict['metadata']['data_loader'], loss_fn=loss_setup(params=params['model_dict']['loss_params'])
                                   , device=params['device_dict']['device'], loss_accum=params['model_dict']['accumulate_loss']),
                    'lambda':0.4,
                    }
    else:
        return None

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

class EWC:
    def __init__(self, model, dataloader, loss_fn,loss_accum, device='cpu'):
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = device
        self.loss_accum = loss_accum

        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher()
    def _compute_fisher(self):
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()

        for data in self.dataloader:
            data = data.to(self.device)
            inputs = data
            self.model.zero_grad()
            output = self.model(inputs)
            output, targets, vec = accumulate_predictions(output, data, self.loss_accum)
            output = output.to(targets.device)
            loss = self.loss_fn(output, targets)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    fisher[n] += p.grad.detach()**2 / len(self.dataloader)
        return fisher
    def get_loss(self,model, base_loss, lambda_ewc=0.4):
        penalty = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                param_diff = p - self.params[n]
                penalty += (self.fisher[n] * param_diff.pow(2)).sum()
        return base_loss + lambda_ewc * penalty













