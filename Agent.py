from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    
    '''
    Abstract base class for agents to play classical games
    '''

    @abstractmethod
    def select_action(self, state, legal_moves):

        '''
        Accept state as arg, return selected action 
        '''

        pass


class RandomAgent(Agent):
    def select_action(self, state, legal_moves):
        return np.random.choice(legal_moves)