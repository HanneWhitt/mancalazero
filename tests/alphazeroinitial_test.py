from mancalazero.agent import AlphaZeroInitial, RandomAgent
from mancalazero.mancala import Mancala
from mancalazero.selfplay import SelfPlay
import time
import numpy as np


state = np.array([1, 1, 0, 1, 0, 0, 21, 0, 0, 0, 1, 0, 0, 23, 0, 40])


mcla = Mancala(state, starting_stones=4)


rev_view = mcla.reverse_view(state)
rev_view[-2] = 1
rev = Mancala(rev_view, starting_stones=4)
print(rev.state)
print(rev.current_player)


azi = AlphaZeroInitial(search_kwargs={'n_sims': 1000})

s = time.time()

for i in range(1):
    p = azi.policy(rev)
    print(p, '\n')

print('Time:', time.time() - s)




sp = SelfPlay(Mancala, 0, game_kwargs={'starting_stones': 4})

agents = [AlphaZeroInitial(), RandomAgent()]


outcomes = sp.tournament(agents, 100)

print(outcomes)