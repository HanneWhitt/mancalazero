from mancalazero.mancalanet import MancalaNet
from mancalazero.loss import AlphaZeroLoss
from mancalazero.gamedataset import GameDataset
from mancalazero.mancala import Mancala
from mancalazero.agent import RandomAgent
from torch.utils.data import DataLoader
from mancalazero.gamedataset import GameDataset, TrainingSampler, custom_collate
import torch

Game = Mancala
agents = [RandomAgent(), RandomAgent()]
game_kwargs = {
    'starting_stones': 4
}
sampler = TrainingSampler(Game, agents, game_kwargs)
dataset = GameDataset(
    sampler,
    positions_per_game=1
)

train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate)

example = next(iter(train_dataloader))

inpt, z, pi, mask = example

v = torch.Tensor([-1])
p = pi.clone()
p[0, 0] = 0
p = p/p.sum()

print('outcome/value')
print('z:', z)
print('v:', v)

print('\nsearch_probs/policy')
print('pi:', pi)
print('p:', p)

print('\nmask')
print(mask)

input('Calc loss?')

cl = AlphaZeroLoss(1)


model = MancalaNet()

loss_value = cl(z, v, pi, p, mask, model.named_parameters())

print('\nloss value')
print(loss_value)



# l2_reg = 0
# for n, m in model.named_parameters():
#     print(n)
#     # print(type(n))
#     print(m.shape)
#     print(m.dtype)
#     input()

    # input()
    # print(m**2)
    # input()
    # print(torch.sum(m**2))
    # print(torch.norm(m)**2)
    # input()
    # l2_reg += torch.sum(m)
    # print(l2_reg)
    # input()



