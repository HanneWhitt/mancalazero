from mancalazero.MancalaNet import MancalaNet
from mancalazero.loss import AlphaZeroLoss
from mancalazero.GameDataset import GameDataset
from mancalazero.mancala import Mancala
from mancalazero.Agent import RandomAgent
from torch.utils.data import DataLoader
from mancalazero.GameDataset import GameDataset, TrainingSampler, custom_collate
import torch


game_kwargs = {
    'starting_stones': 4
}
batch_size = 2048
learning_rate = 0.01
l2_reg_coeff = 0.0001
training_steps = 100


Game = Mancala
agents = [RandomAgent(), RandomAgent()]

sampler = TrainingSampler(Game, agents, game_kwargs)
dataset = GameDataset(
    sampler,
    positions_per_game=1
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)


model = MancalaNet()
loss_fn = AlphaZeroLoss(l2_reg_coeff)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_step(obs, z, pi, mask):

    p, v = model(obs, mask)
    loss = loss_fn(z, v, pi, p, mask, model.named_parameters())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss


model.train()


for step, (obs, z, pi, mask) in enumerate(dataloader):
    
    model.check_all_params()

    print(f'Step {step}')

    loss = train_step(obs, z, pi, mask)

    print('Loss:', loss.item())

    # loss, current = loss.item(), step * batch_size + len(obs)
    # print(f"loss: {loss:>7f}  [{current:>5d}/{training_steps:>5d}]")

    if step >= training_steps:
        break