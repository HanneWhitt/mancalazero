from mancalazero.agent import Agent
from mancalazero.utils import fill, np_nans
from abc import ABC, abstractmethod
import numpy as np
import asyncio


class SelfPlay:

    def __init__(
        self,
        Game,
        agents,
        game_kwargs={},
        exploration_moves=30,
        buffer_size=None
    ):
        self.Game = Game
        self.agents = agents
        self.game_kwargs=game_kwargs
        self.exploration_moves = exploration_moves

        self.buffer_size = buffer_size
        if buffer_size is not None:
            self.outcome_buffer = np_nans(buffer_size, 'int8')
            self.state_buffer = np_nans((buffer_size, *self.Game.shape), 'uint8')
            self.policy_buffer = np_nans((buffer_size, self.Game.total_actions()), 'float32')
            self.mask_buffer = np_nans((buffer_size, self.Game.total_actions()), 'uint8')
            self.filled = 0

    
    def temperature_scheme(self, move_number):
        if move_number < self.exploration_moves:
            return 1
        else:
            return 0


    def play_game(self, outcome_only=False):

        game = self.Game(**self.game_kwargs)
                
        position_record = []
        policy_record = []
        legal_actions_record = []

        while not game.game_over:    

            agent = self.agents[game.current_player]
            temperature = self.temperature_scheme(game.turn_number)
            action, policy = agent.select_action(game, temperature)

            if not outcome_only:
                position_record.append(game.get_observation(game.current_player))
                policy_record.append(policy)
                legal_actions_record.append(game.legal_actions)

            game = game.action(action)

        outcome = game.check_outcome()

        if outcome_only:
            return outcome

        return outcome, position_record, policy_record, legal_actions_record


    def sample_game(self, positions_per_game=1):
        outcome, position_record, policy_record, legal_actions_record = self.play_game()
        n_positions = len(position_record)
        sample_idxs = np.random.choice(n_positions, positions_per_game, replace=False)
        position_sample = [position_record[i] for i in sample_idxs]
        policy_sample = [policy_record[i] for i in sample_idxs]
        legal_actions_sample = [legal_actions_record[i] for i in sample_idxs]
        return outcome, position_sample, policy_sample, legal_actions_sample


    def format(self, outcome, pos, pol, las):

        n_positions = len(pos)

        outcome = np.array(outcome).astype('int8').repeat(n_positions)
        input = np.vstack(pos).astype('uint8')

        search_probs = [fill(self.Game.total_actions(), idxs, probs) for probs, idxs in zip(pol, las)]
        search_probs = np.vstack(search_probs).astype('float32')
        
        masks = [fill(self.Game.total_actions(), la) for la in las]
        masks = np.vstack(masks).astype('uint8')

        return input, outcome, search_probs, masks


    def sample_and_format(self, positions_per_game=1):
        return self.format(*self.sample_game(positions_per_game))



    # def buffer_game(self):

    #     z, s, pi, msk = self.format(*self.play_game())
    #     n_moves = s.shape[0]

    #     overshoot = self.filled + n_moves - self.buffer_size

    #     if overshoot > 0:
    #         self.buffer

    #     outcome_buffer[moves_in_buffer:end] = z
    #     state_buffer[moves_in_buffer:end] = s
    #     policy_buffer[moves_in_buffer:end] = pi
    #     mask_buffer[moves_in_buffer:end] = msk

    # return 

