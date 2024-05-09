from mancalazero.mancalanet import MancalaNet
from mancalazero.loss import AlphaZeroLoss
from mancalazero.gamedataset import GameDataset
from mancalazero.mancala import Mancala
from mancalazero.agent import AlphaZeroAgent, RandomAgent
from torch.utils.data import DataLoader
from mancalazero.gamedataset import GameDataset, custom_collate
import torch
import numpy as np

import time


game_kwargs = {
    'starting_stones': 4
}

batch_size = 2
learning_rate = 0.01
l2_reg_coeff = 0.0001
training_steps = 100


starting_model = MancalaNet()
starting_agent = AlphaZeroAgent(
    starting_model,
    mcts_kwargs={'noise_fraction': 0.25},
    search_kwargs={'n_sims': 800}
)


# board_config = np.array([0, 0, 0, 1, 0, 0, 0, 44, 1, 1, 0, 1, 0, 0, 1, 45])
# example_state = Mancala(board_config, starting_stones=4)
# print(example_state.display())
# input()

# action, policy = starting_agent.select_action(example_state, temperature=1)
# print(action, policy)
# print()

# input()


Game = Mancala



agents = [starting_agent, starting_agent]

# agents = [RandomAgent(), RandomAgent()]

sampler = TrainingSampler(Game, agents, game_kwargs=game_kwargs)
dataset = GameDataset(
    sampler,
    positions_per_game=1
)


dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)#, num_workers=5)


train_model = MancalaNet()
loss_fn = AlphaZeroLoss(l2_reg_coeff)
optimizer = torch.optim.SGD(train_model.parameters(), lr=learning_rate)


def train_step(obs, z, pi, mask):

    p, v = train_model(obs, mask)
    loss = loss_fn(z, v, pi, p, mask, train_model.named_parameters())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss


train_model.train()

s = time.time()

for step, (obs, z, pi, mask) in enumerate(dataloader):

    print(f'Step {step}')
    print('Batch creation time: ', round(time.time() - s, 3))
    s = time.time()

    # train_model.check_all_params()


    loss = train_step(obs, z, pi, mask)

    print('Loss:', loss.item())
    print('Network train time: ',  round(time.time() - s, 3))
    s = time.time()


    # loss, current = loss.item(), step * batch_size + len(obs)
    # print(f"loss: {loss:>7f}  [{current:>5d}/{training_steps:>5d}]")

    if step >= training_steps:
        break