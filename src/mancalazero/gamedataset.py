import torch
from torch.utils.data import Dataset
from mancalazero.selfplay import SelfPlay
import numpy as np
    


def custom_collate(batch):
    batch = [torch.from_numpy(np.vstack(x)) for x in zip(*batch)]
    batch[1] = batch[1].view(-1)
    return batch


class GameDataset(Dataset):

    def __init__(
        self,
        sampler: SelfPlay,
        positions_per_game=1,
        n_examples=10**10
    ):
        self.sampler = sampler
        self.positions_per_game = positions_per_game
        self.n_examples = n_examples


    def __len__(self):
        return self.n_examples
    

    def __getitem__(self, idx):

        print(idx)
        
        idx = idx % self.n_examples
        np.random.seed(idx)

        sample = self.sampler.sample_and_format(self.positions_per_game)

        return [x.astype('float32') for x in sample]