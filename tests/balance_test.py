from mancalazero.mancala import MancalaBoard
from mancalazero.Agent import RandomAgent
import pandas as pd

# Test: play two random agents against each other 10000 times
# Good test for rules implementation
# Also gives info on balance of different variants under random play

results = {}

n_games = 10000

for n_stones in [3, 4]:
    for variant in ['end_game', 'pass_back']:

        game_outcomes = []

        for game in range(n_games):

            if game % 100 == 0:
                print(f'{n_stones} stones, {variant} variant, game {game}/{n_games}')

            mancala_game = MancalaBoard(no_moves_policy=variant, starting_stones=n_stones)

            Agent_0 = RandomAgent()
            Agent_1 = RandomAgent()

            while not mancala_game.game_over:

                #mancala_game.display()

                if mancala_game.current_player == 0:
                    action = Agent_0.select_action(mancala_game.state, mancala_game.legal_actions)
                    #print('Agent 0 chose action: ', action)
                else:
                    action = Agent_1.select_action(mancala_game.state, mancala_game.legal_actions)
                    #print('Agent 1 chose action: ', action)
                mancala_game = mancala_game.action(action)

            player_0_score, player_1_score = mancala_game.check_score()
            outcome = mancala_game.check_outcome()

            game_data = {
                'outcome': outcome,
                'player_0_score': player_0_score,
                'player_1_score': player_1_score,
                'turns_played': mancala_game.turn_number
            }

            game_outcomes.append(game_data)

        game_outcomes = pd.DataFrame(game_outcomes)

        variant_name = f'{variant}_variant_{n_stones}_stones'

        outfile = f'results/balance_tests/{variant_name}.csv'

        game_outcomes.to_csv(outfile)

        results[variant_name] = game_outcomes
