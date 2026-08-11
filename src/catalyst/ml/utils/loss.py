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
    """Apply a base loss to the worst-error fraction of samples.

    Ranking is performed on the first (sample) dimension using mean absolute
    prediction error across any remaining scalar/vector channels.  The selected
    samples retain their original shape when passed to ``sub_function``.
    """
    def __init__(self, percent, sub_function):
        super(MaxNpercent, self).__init__()
        percent = float(percent)
        if not (0.0 < percent <= 1.0):
            raise ValueError("MaxNpercent percent must satisfy 0 < percent <= 1.")
        if not callable(sub_function):
            raise TypeError("MaxNpercent sub_function must be callable.")
        self.percent = percent
        self.sub_function = sub_function

    def forward(self, input, target):
        if input.shape != target.shape:
            raise ValueError(
                "MaxNpercent requires input and target to have identical shapes; "
                f"received {tuple(input.shape)} and {tuple(target.shape)}."
            )
        if input.ndim == 0 or input.shape[0] == 0:
            raise ValueError("MaxNpercent requires at least one sample.")

        n_samples = int(input.shape[0])
        n_select = min(n_samples, max(1, math.ceil(self.percent * n_samples)))
        per_sample_error = torch.abs(input - target)
        if per_sample_error.ndim > 1:
            per_sample_error = per_sample_error.reshape(n_samples, -1).mean(dim=1)

        selected = torch.topk(
            per_sample_error,
            k=n_select,
            largest=True,
            sorted=False,
        ).indices
        return self.sub_function(input.index_select(0, selected), target.index_select(0, selected))

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













