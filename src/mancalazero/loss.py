import torch
from torch import nn


class AlphaZeroLoss(nn.Module):
    
    def __init__(self, l2_reg_coeff=0.0001):
        super(AlphaZeroLoss, self).__init__()
        self.l2_reg_coeff = l2_reg_coeff

    def forward(self, z, v, pi, p, mask, named_parameters):

        # MSE on value estimate vs true outcome
        se = (z - v)**2

        # Safe log of network policy stops zeros (from precision limit) in p from causing -inf in log
        safe_log_p = torch.clamp(torch.log(p), -100)

        # Cross entropy comparing search probs to network policy
        # Invalid moves masked out (so no impact on loss)
        cross_ent = torch.sum(mask*pi*safe_log_p, 1)

        # L2 regularization. Believe PyTorch default applies erroneously to batchnorm parameters and bias?
        l2_reg = 0
        for name, params in named_parameters:
            if 'bn' not in name and 'bias' not in name:
                l2_reg += torch.sum(params**2)

        per_example_loss = se + cross_ent + self.l2_reg_coeff*l2_reg

        return per_example_loss.mean()