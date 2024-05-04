from mancalazero.Agent import Agent
from mancalazero.utils import fill
from abc import ABC, abstractmethod
import numpy as np


class SelfPlay(ABC):

    def __init__(
        self,
        Game,
        agents,
        game_kwargs={}
    ):
        self.Game=Game
        self.agents = agents 
        self.game_kwargs=game_kwargs


    @abstractmethod
    def temperature_scheme(self, move_number):
        
        '''
        Return value for temperature parameter at the given move number
        '''

        pass


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


    def sample_and_format(self, positions_per_game=1):

        outcome, pos, pol, las = self.sample_game(positions_per_game)

        input = np.vstack(pos).astype('float32')
        outcome = np.array(outcome).astype('float32').repeat(positions_per_game)
        
        search_probs = [fill(self.Game.total_actions(), idxs, probs) for probs, idxs in zip(pol, las)]
        search_probs = np.vstack(search_probs).astype('float32')
        
        masks = [fill(self.Game.total_actions(), la) for la in las]
        masks = np.vstack(masks).astype('float32')

        return input, outcome, search_probs, masks


