import torch.multiprocessing as mp
from mancalazero.agent import AlphaZeroAgent, AlphaZeroInitial
from mancalazero.utils import fill, wrap_assign
from mancalazero.gamestate import GameState
from torch import nn
import numpy as np
import time
from itertools import permutations
from collections import Counter


class SelfPlay:

    def __init__(
        self,
        Game: GameState,
        exploration_moves: int,
        Network: nn.Module=None,
        load_buffer_from=None,
        game_kwargs={},
        network_kwargs={},
        agent_kwargs={},
        buffer_size=None,
        n_producers=None
    ):
        self.Game = Game
        self.Network = Network
        self.exploration_moves = exploration_moves
        self.network_kwargs = network_kwargs
        self.agent_kwargs = agent_kwargs
        self.game_kwargs = game_kwargs
        self.n_producers = mp.cpu_count() - 1 if n_producers is None else n_producers

        if load_buffer_from is None:

            self.buffer_size = buffer_size
            if buffer_size is not None:
                
                self.state_buffer = np.zeros((buffer_size, *self.Game.shape), 'uint8')
                self.outcome_buffer = np.zeros(buffer_size, 'int8')
                self.policy_buffer = np.zeros((buffer_size, self.Game.total_actions()), 'float32')
                self.mask_buffer = np.zeros((buffer_size, self.Game.total_actions()), 'uint8')
                self.filled = 0
                self.game_number = 0

        else:
            npz = np.load(load_buffer_from, allow_pickle=True)
            self.buffer_size = int(npz['buffer_size'])
            self.state_buffer = npz['state_buffer']
            self.outcome_buffer = npz['outcome_buffer']
            self.policy_buffer = npz['policy_buffer']
            self.mask_buffer = npz['mask_buffer']
            self.filled = int(npz['filled'])
            self.game_number = int(npz['game_number'])

        if self.buffer_size is not None:
            self.queue = mp.Manager().Queue()


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


    def tournament(self, agents, n_games, all_sides=True, sums=True):
        n_players = len(agents)
        players = list(range(n_players))
        if all_sides:
            per = list(permutations(players))
        else:
            per = [players]
        outcomes = {}
        for order in per:
            outcomes[order] = []
            for g in range(n_games):
                agent_order = [agents[i] for i in order]
                outcome = self.play_game(agent_order, outcome_only=True)
                outcomes[order].append(outcome)
            if sums:
                outcomes[order] = Counter(outcomes[order])
        return outcomes


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

        # start filling the buffer with alphazero version with random prior
        agents = [AlphaZeroInitial(), AlphaZeroInitial()]            

        while True:

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


    def terminate(self):
        for p in self.producers:
            p.kill()
        # self.cons.join()
        print('TERMINATED')


    def sample_from_buffer(self):
        idx = np.random.randint(self.buffer_size)
        input_sample = self.state_buffer[idx, :]
        z_sample = self.outcome_buffer[idx]
        pi_sample = self.policy_buffer[idx, :]
        mask_sample = self.mask_buffer[idx, :]
        return input_sample, z_sample, pi_sample, mask_sample
    

    def save_buffer(self, npzfile):
        np.savez(
            npzfile,
            state_buffer=self.state_buffer,
            outcome_buffer=self.outcome_buffer,
            policy_buffer=self.policy_buffer,
            mask_buffer=self.mask_buffer,
            game_number=self.game_number,
            filled=self.filled,
            buffer_size=self.buffer_size
        )