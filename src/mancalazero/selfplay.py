import torch.multiprocessing as mp
from mancalazero.agent import AlphaZeroAgent, RandomAgent
from mancalazero.utils import fill, wrap_assign
from mancalazero.gamestate import GameState
from torch import nn
import numpy as np
import time


class SelfPlay:

    def __init__(
        self,
        Game: GameState,
        Network: nn.Module,
        exploration_moves,
        game_kwargs={},
        network_kwargs={},
        agent_kwargs={},
        buffer_size=None,
        n_producers=None,
        start_at_game=0,
        max_games=None
    ):
        self.Game = Game
        self.game_kwargs = game_kwargs
        self.Network = Network
        self.exploration_moves = exploration_moves
        self.network_kwargs = network_kwargs
        self.agent_kwargs = agent_kwargs
        self.game_kwargs = game_kwargs

        self.buffer_size = buffer_size
        if buffer_size is not None:
            self.queue = mp.Manager().Queue()
            self.state_buffer = np.zeros((buffer_size, *self.Game.shape), 'uint8')
            self.outcome_buffer = np.zeros(buffer_size, 'int8')
            self.policy_buffer = np.zeros((buffer_size, self.Game.total_actions()), 'float32')
            self.mask_buffer = np.zeros((buffer_size, self.Game.total_actions()), 'uint8')
            self.filled = 0
            self.game_number = start_at_game

        self.n_producers = mp.cpu_count() - 1 if n_producers is None else n_producers
        self.max_games = 10**10 if max_games is None else max_games

    
    def temperature_scheme(self, move_number):
        if move_number < self.exploration_moves:
            return 1
        else:
            return 0


    def play_game(self, agents, outcome_only=False):

        game = self.Game(**self.game_kwargs)
                
        position_record = []
        policy_record = []
        legal_actions_record = []

        while not game.game_over:    

            agent = agents[game.current_player]
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

        return position_record, outcome, policy_record, legal_actions_record


    def sample_game(self, positions_per_game=1):
        outcome, position_record, policy_record, legal_actions_record = self.play_game()
        n_positions = len(position_record)
        sample_idxs = np.random.choice(n_positions, positions_per_game, replace=False)
        position_sample = [position_record[i] for i in sample_idxs]
        policy_sample = [policy_record[i] for i in sample_idxs]
        legal_actions_sample = [legal_actions_record[i] for i in sample_idxs]
        return position_sample, outcome, policy_sample, legal_actions_sample


    def format(self, pos, outcome, pol, las):
        n_positions = len(pos)
        input = np.vstack(pos).astype('uint8')
        outcome = np.array(outcome).astype('int8').repeat(n_positions)
        search_probs = [fill(self.Game.total_actions(), idxs, probs) for probs, idxs in zip(pol, las)]
        search_probs = np.vstack(search_probs).astype('float32')
        masks = [fill(self.Game.total_actions(), la) for la in las]
        masks = np.vstack(masks).astype('uint8')
        return input, outcome, search_probs, masks


    def sample_and_format(self, positions_per_game=1):
        return self.format(*self.sample_game(positions_per_game))


    def add_to_buffer(self, pos, outcome, pol, las):
        n_actions = pos.shape[0]
        start = self.filled % self.buffer_size
        end = (self.filled + n_actions) % self.buffer_size
        wrap_assign(self.state_buffer, pos, start, end)
        wrap_assign(self.outcome_buffer, outcome, start, end)
        wrap_assign(self.policy_buffer, pol, start, end)
        wrap_assign(self.mask_buffer, las, start, end)
        self.game_number += 1
        self.filled += n_actions


    def producer(self, prod_idx, conn):

        item_idx = self.game_number + prod_idx 

        # start filling the buffer with random games
        # TODO: replace with random-prior MCTS
        agents = [RandomAgent(), RandomAgent()]            

        while item_idx < self.max_games:

            #print('Generating game: ', item_idx)

            # Generate a new random game every time
            np.random.seed(item_idx)

            s = time.time()

            message_available = conn.poll()

            if message_available:
                
                print('MESSAGE RECEIVED')
                latest_weights = conn.recv()
                net = self.Network(**self.network_kwargs)
                net.load_state_dict(latest_weights)
                agent = AlphaZeroAgent(net, **self.agent_kwargs)
                agents = [agent, agent]
                print('WEIGHTS UPDATED')

            result = self.format(*self.play_game(agents))
            
            self.queue.put(result)
            item_idx += self.n_producers
    
            #print(f'Game {item_idx} generated in ', round(time.time() - s, 2))


    def consumer(self):
        added = 0
        st = time.time()
        while not self.queue.empty():
            game_record = self.queue.get()
            self.add_to_buffer(*game_record)
            added += 1
        total_time = time.time() - st
        print(f'ADDED: {added}, time {total_time}')
        if added > 0:
            av_t = total_time/added
            print('Average per element: ', av_t)
            

    def message_producers(self, message):
        for parent_conn, child_conn in self.connections:
            #We only want most recent, so fuyirst empty message buffer
            while child_conn.poll():
                child_conn.recv()
            parent_conn.send(message)


    def run_buffer(self, start_message=None):
        self.connections = [mp.Pipe() for i in range(self.n_producers)]
        self.producers = [mp.Process(target=self.producer, args=(i, conn[1])) for i, conn in enumerate(self.connections)]
        if start_message is not None:
            self.message_producers(start_message)
            time.sleep(1)
        for p in self.producers:
            p.start()
        #self.cons = mp.Process(mp.Process(target=self.producer))


    def terminate(self):
        for p in self.producers:
            p.kill()
        # self.cons.join()
        print('TERMINATED')





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

