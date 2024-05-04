from abc import ABC, abstractmethod
from mancalazero.Game import GameState
import numpy as np


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