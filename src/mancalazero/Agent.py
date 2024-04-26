from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    
    '''
    Abstract base class for agents to play classical games
    '''

    @abstractmethod
    def policy(self, state, legal_actions):
        '''
        Return a probability of selecting each move according to the agent's judgement 

        Must only return policy over legally valid moves
        '''
        pass
        

    def select_action(self, state, legal_actions, temperature=1):
        
        '''
        Accept state as arg, return selected action 
        '''
        
        policy = self.policy(state, legal_actions)
        
        # If we want the best possible choice, t=0 i.e zero out all the non-maximal elements of the policy
        if temperature==0:
            max_policy = policy.max()
            policy[policy != max_policy] = 0
            policy = policy/policy.sum()
        elif temperature != 1:
            raise NotImplementedError('temp params not 0 (infintesimal) or 1 not implemented yet')

        # Even if non-stochastic, it's possible there could be more than one action with max policy
        # Select randomly from these
        return np.random.choice(legal_actions, p=policy)


class RandomAgent(Agent):

    def policy(self, state, legal_actions):
        n_legal_actions = len(legal_actions)
        uniform_p = 1/n_legal_actions
        policy = np.ones(n_legal_actions)*uniform_p
        return policy