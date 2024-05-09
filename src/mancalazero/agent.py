from abc import ABC, abstractmethod
from mancalazero.gamestate import GameState
from mancalazero.mcts import MCTSNode
from mancalazero.utils import fill
import numpy as np
import torch


class Agent(ABC):
    
    '''
    Abstract base class for agents to play classical games
    '''

    @abstractmethod
    def policy(self, state: GameState):
        '''
        Return a probability of selecting each move according to the agent's judgement 

        Must only return probability over legal moves
        '''
        pass
        

    def select_action(self, state:GameState, temperature=1):
        
        '''
        Accept state as arg, return selected action 
        '''
        
        policy = self.policy(state)
        
        # If we want the best possible choice, t=0 i.e zero out all the non-maximal elements of the policy
        if temperature==0:
            max_policy = policy.max()
            policy[policy != max_policy] = 0
            policy = policy/policy.sum()
        elif temperature != 1:
            raise NotImplementedError('temp params not 0 (infintesimal) or 1 not implemented yet')

        # Even if non-stochastic, it's possible there could be more than one action with max policy
        # Select randomly from these
        action = np.random.choice(state.legal_actions, p=policy)

        return action, policy


class RandomAgent(Agent):

    def policy(self, state):
        legal_actions = state.legal_actions
        n_legal_actions = len(legal_actions)
        uniform_p = 1/n_legal_actions
        policy = np.ones(n_legal_actions)*uniform_p
        return policy
    

class AlphaZeroAgent(Agent):

    def __init__(
        self,
        network,
        mcts_kwargs={},
        search_kwargs={}
    ):
        self.network = network
        self.mcts_kwargs = mcts_kwargs
        self.search_kwargs=search_kwargs


    def policy(self, state: GameState):
        mcts = MCTSNode(
            state,
            self.network_prior,
            **self.mcts_kwargs
        )
        search_probabilities = mcts.search(**self.search_kwargs)
        return search_probabilities
    

    def network_prior(self, state: GameState):

        """
        Just a wrapper for torch policy/value model; puts in evaluation mode, adapts torch.tensor/numpy

        Evaluate the policy/value network for a single gamestate

        TODO: introduce queueing/batch evaluation
        """

        observation = state.get_observation(state.current_player)

        observation = torch.from_numpy(observation).float()
        observation = observation[None, :]

        legal_actions = state.legal_actions
        mask = fill(state.total_actions(), legal_actions)
        mask = torch.from_numpy(mask).float()
        mask = mask[None, :]

        self.network.eval()
        p, v = self.network(observation, mask)

        return p.detach().numpy()[0][legal_actions], v.detach().numpy()[0][0]