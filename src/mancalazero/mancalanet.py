import torch
from torch import nn
import torch.nn.functional as F
    

class MancalaNet(nn.Module):

    '''
    A neural network to take as input:
    - mancala board state vector s
    - mancala board binary legal moves mask (len 6)
    And output: 
    - a policy vector p (len 6) 
    - scalar estimated value v

    Main body is a simple MLP with batch normalisation, effectively produces a board embedding. 
    
    Both policy and value head then have a layer or two to themselves to enable some specialisation in the way they use 
    the shared board embedding. Imitates approach in much bigger AlphaZero CNN

    Note batchnorm applied AFTER activation function in all cases... Not done in original paper but experiments since seem to 
    support this approach https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
    '''

    def __init__(self,
        input_length = 16,
        shared_layers = [32, 32],
        policy_head_layers = [32],
        value_head_layers = [32],
        n_actions = 6
    ):
        
        super(MancalaNet, self).__init__()

        shared_dims = [input_length, *shared_layers]
        self.core = nn.ModuleList([nn.Linear(i, j) for i, j in zip(shared_dims, shared_layers)])
        self.core_bn = nn.ModuleList([nn.BatchNorm1d(j) for j in shared_layers])

        policy_dims = [shared_layers[-1], *policy_head_layers, n_actions]
        self.p_head = nn.ModuleList([nn.Linear(i, j) for i, j in zip(policy_dims, policy_dims[1:])])
        self.p_bn = nn.ModuleList([nn.BatchNorm1d(j) for j in policy_head_layers])

        value_dims = [shared_layers[-1], *value_head_layers, 1]
        self.v_head = nn.ModuleList([nn.Linear(i, j) for i, j in zip(value_dims, value_dims[1:])])
        self.v_bn = nn.ModuleList([nn.BatchNorm1d(j) for j in value_head_layers])


    def masked_softmax(self, input, mask):

        """
        We need to manually implement a version of softmax that ignores zeros in the mask; i.e 
        it outputs a probability distribution with zeros for illegal moves, and probabilities summing
        to 1 for all other moves. Exp(0) = 1, so this is not as simple as multiplying input by mask.
        """
        # Take exponential for all values 
        exp = torch.exp(input)
    
        # THEN apply mask
        masked = exp*mask

        # Then apply softmax
        return masked/masked.sum(1, keepdim=True)


    def forward(self, obs, mask):

        # Core
        for layer, batchnorm in zip(self.core, self.core_bn):
            obs = layer(obs)
            obs = F.relu(obs)
            obs = batchnorm(obs)

        # Policy head
        p = obs.clone()
        for p_layer, batchnorm in zip(self.p_head, self.p_bn):
            p = p_layer(p)
            p = F.relu(p)
            p = batchnorm(p)

        # No batchnorm on last layer
        p = self.p_head[-1](p)

        # Softmax for valid probability distribution, mask to zero out illegal moves
        p = self.masked_softmax(p, mask)

        # Value head
        v = obs.clone()
        for v_layer, batchnorm in zip(self.v_head, self.v_bn):
            v = v_layer(v)
            v = F.relu(v)
            v = batchnorm(v)

        # No batchnorm on last layer
        v = self.v_head[-1](v)

        # tanh on value to match outcome to range [-1, 1]
        v = F.tanh(v)

        return p, v


    def check_all_params(self):
        problem = False
        for name, tensor in self.named_parameters():
            problem = check_nan_inf(tensor, name)
        if not problem:
            print('check_all_params ran, no problem found')


def check_nan_inf(tensor, name):
    nan_inf = False
    if torch.isnan(tensor).any():
        print(f"Tensor '{name}' features nan values")
        response = input("Print tensor?")
        if response == '':
            print(tensor)
        input()
        nan_inf = True
    elif not torch.isfinite(tensor).all():
        print(f"Tensor '{name}' features inf values")
        response = input("Print tensor?")
        if response == '':
            print(tensor)
        input()
        nan_inf = True
    return nan_inf