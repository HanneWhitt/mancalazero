from mancalazero.agent_play import agent_play
from mancalazero.mancala import MancalaBoard
from mancalazero.Agent import RandomAgent


Game = MancalaBoard
agents = [RandomAgent(), RandomAgent()]
n_games = 3
game_kwargs = {
    'starting_stones': 4
}
records_per_game=2

res = agent_play(
    Game,
    agents,
    n_games,
    game_kwargs=game_kwargs,
    records_per_game=records_per_game
)

outcomes, positions = res 

print(positions)