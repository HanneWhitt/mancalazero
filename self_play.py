from MCTS import MCTS
from policy_and_value_network import PolicyAndValue
from mancala import MancalaBoard
import tensorflow as tf


class MancalaMCTS(MCTS):

    def __init__(self, 
                state: MancalaBoard,
                network: PolicyAndValue,
                noise_fraction=0, 
                dirichlet_alpha=3, 
                distribution_validity_epsilon=1e-6):

        super(MancalaMCTS, self).__init__(noise_fraction, dirichlet_alpha, distribution_validity_epsilon)

        self.network = network


    def prior_function(self, state_representation, legal_moves):
        
        '''
        We can do MCTS for any game by overriding prior_function from base MCTS class. Later project - chess agent!
        '''

        legal_moves_one_hot_encoded = np.zeros(6)
        legal_moves_one_hot_encoded[legal_moves] = 1
        p, v = self.network(state_representation, legal_moves_one_hot_encoded)

        return p, v


def agent_v_agent_game(board, agent_1, agent_2=None, max_move_time=None, max_move_sims=None):

    '''
    Implementing self-play (same network) and evaluation play (different networks).

    Board should be an instance of an object like MancalaBoard, with the functions listed in MCTS.py.

    agent_1 should be an instance of a class inheriting from MCTS base class, like MancalaMCTS.

    agent_2 is an optional extra instance of such a class: 
    If it is subbmitted, the agent v. agent game will be run in evaluation mode, one against the other.
    Else, the function carries out a self-play game to produce training data.
    '''

    game_history = []

    while not board.game_over():

        best_move, MCTS = agent_1.build_tree_and_choose_move()
        

        if agent_2:
