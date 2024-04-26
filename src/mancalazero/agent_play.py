from mancalazero.Agent import Agent
import numpy as np


def default_temp_scheme(move_number):
    if move_number < 30:
        return 1
    else:
        return 0


def agent_play(
    Game,
    agents,
    n_games,
    game_kwargs={},
    records_per_game=0,
    temperature_scheme=default_temp_scheme
):

    outcomes = np_nans(n_games)

    game = Game(**game_kwargs)

    if records_per_game:
        total_records = n_games*records_per_game
        obs_shape = game.get_observation(game.current_player).shape
        positions = np_nans((total_records, *obs_shape))
            
    for game_idx in range(n_games):

        if game_idx % 100 == 0:
            print(f'Game {game_idx}/{n_games}')
        
        game_record = []

        while not game.game_over:    

            agent = agents[game.current_player]
            observation = game.get_observation(game.current_player)
            legal_actions = game.legal_actions

            if records_per_game:
                game_record.append(observation)
            
            temperature = temperature_scheme(game.turn_number)
            action = agent.select_action(observation, legal_actions, temperature)
            game = game.action(action)

        outcomes[game_idx] = game.check_outcome()

        if records_per_game:
            game_record = np.vstack(game_record)
            n_actions = game_record.shape[0]
            sample_idx = np.random.choice(n_actions, records_per_game, replace=False)
            start_idx = game_idx*records_per_game
            end_idx = start_idx + records_per_game
            positions[start_idx:end_idx] = game_record[sample_idx]

        game = Game(**game_kwargs)

    if records_per_game:
        outcomes = np.repeat(outcomes, records_per_game)
        return outcomes, positions
    else:
        return outcomes


def np_nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a