from mancalazero.mancala import MancalaBoard
from mancalazero.MCTS import MCTSNode
import numpy as np
from mancalazero.visualization import MCTS_visualization, MCTS_expansion_series


class TestMCTSNode(MCTSNode):

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
        N_parent = self.N.sum() + 1
        new = {
            'state': boardview,
            'N_parent': N_parent,
            'C_puct': self.C_puct(N_parent),
            'U': self.U().tolist(),
            'score': self.action_scores().tolist()
        }
        return {**new, **original}
    

    # def get_edge_description(self, child_idx):
    #     original = super().get_edge_description(child_idx)
    #     new = {
    #         'U': self.U()[child_idx],
    #         'score': self.action_scores()[child_idx]
    #     }
    #     return {**original, **new}


mancala_game_state = MancalaBoard(starting_stones=4)
np.random.seed(0)
test_mcts_node = TestMCTSNode(
    mancala_game_state,
    c_init=2,
    noise_fraction=0.5,
)
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
    savefolder='scratch/examining_U_etc',
    node_label_keys=['state', 'V', 'Q', 'P', 'U', 'score'],
    figsize=(25, 9),
    node_size=6500,
)


print(test_mcts_node.search_probabilities())




# mancala_game_state = MancalaBoard(starting_stones=4)
# np.random.seed(0)

# test_mcts_node = TestMCTSNode(mancala_game_state, c_init=2)
# search_probs = test_mcts_node.search(10)

# print(search_probs)

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