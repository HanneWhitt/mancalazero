from MCTS import MCTS
from policy_and_value_network import PolicyAndValue
from mancala import MancalaBoard
from Agent import RandomAgent


def agent_v_agent_game(board, agent_1, agent_2):

    '''
    Two agents play against each other until game ends.

    board should be an instance of a class inheriting from the Game base class.

    agent_1 and agent_2 should be instances of classes inheriting from the Agent base class.
    '''

    game_history = []

    while not board.game_over():

        best_move, MCTS = agent_1.build_tree_and_choose_move()
        
        if agent_2:



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