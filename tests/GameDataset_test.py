from mancalazero.GameDataset import GameDataset
from mancalazero.mancala import Mancala
from mancalazero.Agent import RandomAgent
from torch.utils.data import DataLoader
from mancalazero.GameDataset import GameDataset, TrainingSampler, custom_collate
import time


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

# for i in range(2):
#     item = dataset.__getitem__(i)
#     for thing in item:
#         print(thing)
#     input()



train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate)

s = time.time()
example = next(iter(train_dataloader))
#print('Time', time.time() - s)

for thing in example:
    print(thing)
    print(thing.shape)

# train_features = example[0]
# print(train_features.view(-1, 16))