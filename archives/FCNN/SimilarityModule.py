
import torch
import torch.utils.data
import numpy as np

from Experiment import Deficit

class SimilaritySampler(torch.utils.data.Sampler):
    def __init__(self, subset_percentage, quantiles, duration, type='similarity', shuffle=True):
        self.quantiles = quantiles
        self.current_epoch = 0
        self.duration = duration
        self.type = type
        self.shuffle = shuffle
        self.all_indices = np.arange(0, quantiles.shape[0])

        if self.type == 'similarity' :
            mask = quantiles <= subset_percentage
        elif self.type == 'disimilarity':
            mask = quantiles >= 1 - subset_percentage

        self.subset = self.all_indices[mask]
        print(f'subset: {self.subset}')

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __iter__(self):
        if self.current_epoch < self.duration :
            indices = self.subset
        else :
            indices = self.all_indices.copy()

        if self.shuffle :
            indices = indices[torch.randperm(indices.shape[0])]
        

        return iter(indices)


class SimilarityTypeDeficit(Deficit):
    def __init__(self, subset_size, quantiles, duration, type='similarity', shuffle=True):
        self.subset_size = subset_size
        self.quantiles = quantiles
        self.duration = duration
        self.type = type
        self.shuffle = shuffle

        self.sampler = SimilaritySampler(self.subset_size, self.quantiles, self.duration, type=self.type, shuffle=self.shuffle)

    
    def update_deficit(self, epoch):
        self.sampler.set_epoch(epoch)
