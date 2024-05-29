import torch
from torch import nn


class AlphaZeroLoss(nn.Module):
    
    def __init__(self, l2_reg_coeff=0.0001, eps=2.718281828459045**-70):
        super(AlphaZeroLoss, self).__init__()
        self.l2_reg_coeff = l2_reg_coeff
        self.eps = eps

    def forward(self, z, v, pi, p, named_parameters):

        # MSE on value estimate vs true outcome
        per_example_loss = (z - v)**2

        # # Safe log of network policy stops zeros (from precision limit) in p from causing -inf in log
        safe_log_p = torch.log(p + self.eps)

        # Cross entropy comparing search probs to network policy
        # Invalid moves masked out (so no impact on loss)
        per_example_loss -= torch.sum(pi*safe_log_p, dim=1)

        # L2 regularization. Believe PyTorch default applies erroneously to batchnorm parameters and bias?
        # for name, params in named_parameters:
        #     if 'bn' not in name and 'bias' not in name:
        #         per_example_loss += self.l2_reg_coeff*torch.sum(params**2)

        return per_example_loss.mean()