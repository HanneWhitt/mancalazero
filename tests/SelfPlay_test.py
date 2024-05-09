from mancalazero.selfplay import SelfPlay
from mancalazero.mancala import Mancala
from mancalazero.agent import RandomAgent


Game = Mancala
agents = [RandomAgent(), RandomAgent()]
game_kwargs = {
    'starting_stones': 4
}
n_games = 3

class TrainingGame(SelfPlay):
    def temperature_scheme(self, move_number):
        if move_number < 30:
            return 1
        else:
            return 0

sp = TrainingGame(Game, agents, game_kwargs)

res = sp.sample_game()

outcomes, policies, positions, legal_actions = res 

print(outcomes)
print(policies)
print(positions)
print(legal_actions)
