from mancalazero.mancala import MancalaBoard
from mancalazero.MCTS import MCTSNode
import numpy as np
from mancalazero.visualization import MCTS_visualization, MCTS_expansion_series


class TestMCTSNode(MCTSNode):

    np.random.seed(0)

    def prior_function(self, observation, legal_actions):

        """
        A dumb prior function for testing.

        Returns:
        
        (i) a policy which is a uniform probability distribution across all the legal moves
        (ii) A value which is a random number in range -1 to 1
        
        """

        n_legal_actions = len(legal_actions)
        uniform_p = 1/n_legal_actions

        policy = np.ones(n_legal_actions)*uniform_p
        value = np.random.uniform(low=-1.0, high=1.0)

        return policy, value


    def get_node_description(self):
        original = super().get_node_description()
        boardview = self.state.display()
        new = {'state': boardview}
        return {**new, **original}


mancala_game_state = MancalaBoard(starting_stones=4)
test_mcts_node = TestMCTSNode(mancala_game_state)
nodes_edges_list = [test_mcts_node.get_nodes_and_edges()]


for i in range(10):
    test_mcts_node.simulation()
    nodes_edges_list.append(test_mcts_node.get_nodes_and_edges())



# MCTS_visualization(
#     *nodes_edges_list[-1],
#     node_label_keys=['state', 'V'],
#     savefile='scratch/test_viz.jpeg',
#     title='Test2',
#     figsize=(12, 7)
# )
MCTS_expansion_series(
    nodes_edges_list,
    savefolder='scratch/expansion_visualisation_test',
    node_label_keys=['state', 'V'],
    figsize=(12, 8)
)





# def C_puct(N_parent, c_init=1.25, c_base=19652):

#     '''
#     A multiplying factor on the confidence bound.
#     '''

#     C = np.log((1 + N_parent + c_base)/c_base) + c_init

#     return C

# N = np.array([0, 1, 0, 2, 0, 3])
# P = np.array([0, 0.1, 0.3, 0.4, 0, 0.2])
# C_puct = 1.25
# N_parent = 6

# U = C_puct * P * np.sqrt(N_parent) / (1 + N)

# print(U)

# noise = dirichlet_noise(3, 3)

# print('\n\n\n', noise, np.sum(noise))