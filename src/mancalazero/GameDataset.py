import torch
from torch.utils.data import Dataset
from mancalazero.SelfPlay import SelfPlay
import numpy as np


class TrainingSampler(SelfPlay):
    def temperature_scheme(self, move_number):
        if move_number < 30:
            return 1
        else:
            return 0


def custom_collate(batch):
    batch = [torch.from_numpy(np.vstack(x)) for x in zip(*batch)]
    batch[1] = batch[1].view(-1)
    return batch


class GameDataset(Dataset):

    def __init__(
        self,
        sampler: TrainingSampler,
        positions_per_game=1,
        n_examples=10**10
    ):
        self.sampler = sampler
        self.positions_per_game = positions_per_game
        self.n_examples = n_examples


    def __len__(self):
        return self.n_examples
    

    def __getitem__(self, idx):
        
        idx = idx % self.n_examples
        np.random.seed(idx)

        return self.sampler.sample_and_format(self.positions_per_game)