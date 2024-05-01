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
        self.shared = [(nn.Linear(i, j), nn.BatchNorm1d(j)) for i, j in zip(shared_dims, shared_dims[1:])]

        policy_dims = [shared_layers[-1], *policy_head_layers, n_actions]
        self.p_layers = [(nn.Linear(i, j), nn.BatchNorm1d(j)) for i, j in zip(policy_dims, policy_dims[1:])]

        value_dims = [shared_layers[-1], *value_head_layers, 1]
        self.v_layers = [(nn.Linear(i, j), nn.BatchNorm1d(j)) for i, j in zip(value_dims, value_dims[1:])]


    def masked_softmax(self, input, mask):

        """
        We need to manually implement a version of softmax that ignores zeros in the mask; i.e 
        it outputs a probability distribution with zeros for illegal moves, and probabilities summing
        to 1 for all other moves. Exp(0) = 1, so this is not as simple as multiplying input by mask.
        """

        print('\n input\n', input)

        # Take exponential for all values 
        exp = torch.exp(input)

        print('\n exp\n',exp)
    
        # THEN apply mask
        masked = exp*mask

        print('\n masked\n', masked)

        # Then apply softmax
        return masked/masked.sum(1, keepdim=True)


    def forward(self, s, mask):
        
        for layer, batchnorm in self.shared:
            s = layer(s)
            s = F.relu(s)
            s = batchnorm(s)

        p = s
        for p_layer, batchnorm in self.p_layers[:-1]:
            p = p_layer(p)
            p = F.relu(p)
            p = batchnorm(p)
        p = self.p_layers[-1][0](p)

        # Softmax for valid probability distribution, mask to zero out illegal moves
        p = self.masked_softmax(p, mask)

        v = s
        for v_layer, batchnorm in self.v_layers[:-1]:
            v = v_layer(v)
            v = F.relu(v)
            v = batchnorm(v)
        v = self.v_layers[-1][0](v)

        # tanh on value to match outcome in range [-1, 1]
        v = F.tanh(v)

        return p, v


