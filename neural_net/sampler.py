import os
import torch
import random
import numpy as np

from torch.utils.data.sampler import Sampler

class ShuffleSampler(Sampler):
    def __init__(self, data_source, seed):
        super().__init__(data_source)
        self.data_source = data_source
        self.seed = seed

        self.index_list = list(range(len(data_source)))
        self.r = random.Random(seed)
        return

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        self.r.shuffle(self.index_list)
        for x in self.index_list:
            yield x