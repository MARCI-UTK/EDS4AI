import torch
import torch.utils.data
import numpy as np

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


from torch.utils.data import TensorDataset, DataLoader
if __name__ == "__main__" :
    torch.manual_seed(0)
    data = torch.from_numpy(np.array([30, 40, 50, 60, 70]))
    labels = torch.from_numpy(np.array([1, 0, 0, 1, 1]))
    quantiles = torch.from_numpy(np.array([0.3, 0.8, 0.8, 0.3, 0.3]))

    dataset = TensorDataset(data, labels)

    sampler = SimilaritySampler(0.5, quantiles, 3, 'similarity')
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=4)

    print('here')

    for epoch in range(10):
        sampler.set_epoch(epoch)

        print(f'epoch: {epoch}')
        for i, (val, label) in enumerate(dataloader):
            print(f'{i}: val: {val}')