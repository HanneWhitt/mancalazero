from abc import ABC, abstractmethod
from mancalazero.gamestate import GameState
from mancalazero.mcts import MCTSNode
from mancalazero.utils import fill, add_dirichlet_noise
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

    def __init__(self, noise_fraction=0, alpha=3, seed=0):
        self.noise_fraction = noise_fraction
        self.alpha = alpha
        self.seed = seed

    def policy(self, state):
        legal_actions = state.legal_actions
        n_legal_actions = len(legal_actions)
        uniform_p = 1/n_legal_actions
        policy = np.ones(n_legal_actions)*uniform_p
        np.random.seed(self.seed)
        self.seed += 1
        policy = add_dirichlet_noise(policy, self.alpha, self.noise_fraction)
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
            self.prior_function,
            **self.mcts_kwargs
        )
        search_probabilities = mcts.search(**self.search_kwargs)
        return search_probabilities
    

    def prior_function(self, state: GameState):

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
        p, v = self.network(observation)

        p = p.detach().numpy()[0][legal_actions]
        p = p/p.sum()

        v = v.detach().numpy()[0]

        return p, v
    


class AlphaZeroInitial(AlphaZeroAgent):

    """
    A version of AlphaZero that uses no network, roughly equivalent to initial untrained AlphaZeroAgent

    Useful for initialisation as it runs faster, avoiding unnecessary evaluations of untrained p/v network

    Provide some noise to encourage exploration
    """

    def __init__(
        self,
        mcts_kwargs={},
        search_kwargs={}
    ):
        super().__init__(None, mcts_kwargs, search_kwargs)

    
    def prior_function(self, state: GameState):
        
        # Uniform distribution with a little dirichlet noise
        legal_actions = state.legal_actions
        n_legal_actions = len(legal_actions)
        uniform_p = 1/n_legal_actions
        p = np.ones(n_legal_actions)*uniform_p
        p = add_dirichlet_noise(p, 3, 0.25)

        # Value frwn from narrow normal distribution, clipped to range -0.5, 0.5
        v = np.clip(np.random.normal(scale=0.15), -0.5, 0.5)

        return p, v
