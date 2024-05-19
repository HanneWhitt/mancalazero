from torch.utils.data import Dataset
from mancalazero.selfplay import SelfPlay
import numpy as np


class GameDataset(Dataset):

    def __init__(
        self,
        sampler: SelfPlay,
        n_examples=10**100
    ):
        self.sampler = sampler
        self.n_examples = n_examples


    def __len__(self):
        return self.n_examples
    

    def __getitem__(self, idx):
        
        np.random.seed(idx)
        sample = self.sampler.sample_from_buffer()

        return [x.astype('float32') for x in sample]
