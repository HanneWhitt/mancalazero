import torch
from torch import nn
import torch.nn.functional as F


class PolicyValueFeedForward(nn.Module):


    def __init__(self,
        input_length = 16,
        shared_layers = [32, 32],
        policy_head_layers = [32],
        value_head_layers = [32],
        n_actions = 6
    ):
        
        super(PolicyValueFeedForward, self).__init__()

        shared_dims = [input_length, *shared_layers]
        self.shared_layers = [nn.Linear(i, j) for i, j in zip(shared_dims, shared_dims[1:])]

        policy_dims = [shared_layers[-1], *policy_head_layers, n_actions]
        self.policy_layers = [nn.Linear(i, j) for i, j in zip(policy_dims, policy_dims[1:])]

        value_dims = [shared_layers[-1], *value_head_layers, 1]
        self.value_layers = [nn.Linear(i, j) for i, j in zip(value_dims, value_dims[1:])]


    def forward(self, representation, legal_moves_mask):
        
        for lyr in self.shared_layers:
            representation = F.relu(lyr(representation))

        policy = representation
        for pol_lyr in self.policy_layers[:-1]:
            policy = F.relu(pol_lyr(policy))
        policy = self.policy_layers[-1](policy)
        # Multiply by legal_moves_mask to set illegal moves to zero probability
        policy = policy*legal_moves_mask
        policy = F.softmax(policy)

        value = representation
        for val_lyr in self.value_layers[:-1]:
            value = F.relu(val_lyr(value))
        value = F.tanh(self.value_layers[-1](value))

        return policy, value


if __name__ == '__main__':

    mnet = PolicyValueFeedForward()
    # print(mnet.shared_layers)
    # print(mnet.policy_layers)
    # print(mnet.value_layers)

    rep = torch.Tensor([1]*16)

    mnet(rep, 1)