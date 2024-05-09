from mancalazero.mancala import Mancala
from mancalazero.mcts import MCTSNode
import numpy as np
from mancalazero.visualisation import MCTS_visualization, MCTS_expansion_series
from mancalazero.gamestate import GameState


def random_prior(state:GameState):

    """
    A dumb prior function for testing.

    Returns:
    
    (i) a policy which is a uniform probability distribution across all the legal moves
    (ii) A value which is a random number in range -1 to 1
    
    """

    n_legal_actions = len(state.legal_actions)
    uniform_p = 1/n_legal_actions

    policy = np.ones(n_legal_actions)*uniform_p
    value = np.random.uniform(low=-1.0, high=1.0)

    return policy, value


mancala_game_state = Mancala(starting_stones=4)

# np.random.seed(0)
test_mcts_node = MCTSNode(
    mancala_game_state,
    prior_function=random_prior,
    c_init=2,
    noise_fraction=0,
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
    savefolder='scratch/opposing_play',
    node_label_keys=['current_player', 'state', 'V', 'Q', 'P', 'U', 'score'],
    edge_label_keys=['action'],
    figsize=(25, 9),
    node_size=6500,
)


print(test_mcts_node.search_probabilities())




# mancala_game_state = Mancala(starting_stones=4)
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