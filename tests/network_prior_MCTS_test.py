from mancalazero.MCTS import MCTSNode
from mancalazero.network_prior import NetworkPrior
from mancalazero.MancalaNet import MancalaNet
from mancalazero.mancala import Mancala


state = Mancala(starting_stones=4)
mancala_net = MancalaNet()
network_prior = NetworkPrior(mancala_net)
mcts = MCTSNode(
    state,
    network_prior
)


print(state.legal_actions)


p, v = network_prior(state)

print(p) 
print(p.sum())
print(v)



pi = mcts.search(100000, 1000)

print(pi)