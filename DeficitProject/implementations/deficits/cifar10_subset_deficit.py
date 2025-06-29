import torch
import torchvision
import numpy as np

from torch.utils.data import random_split, DataLoader
from torchvision.transforms import v2

from exp_driver.deficit import Deficit


class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, start_epoch, end_epoch, subset_size, dataset_size, shuffle=True):
        self.current_epoch = 0
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.all_indices = np.arange(0, dataset_size)
        self.shuffle = shuffle
        self.subset_size = subset_size
        self.dataset_size = dataset_size


    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __iter__(self):
        if self.current_epoch >= self.start_epoch and self.current_epoch < self.end_epoch:
            shuffled = self.all_indices[torch.randperm(self.all_indices.shape[0])]
            indices = shuffled[: int(self.subset_size * self.dataset_size)]
        else :
            indices = self.all_indices.copy()

        if self.shuffle :
            indices = indices[torch.randperm(indices.shape[0])]
        

        return iter(indices)

class CIFAR10SubsetDeficit(Deficit):
    def __init__ (self, deficit_params):
        self.deficit_params = deficit_params
        self.subset_size = self.deficit_params['subset_size']

        if self.deficit_params['dataset'] == 'CIFAR10':
            for param in ['start_epoch', 'end_epoch', 'subset_size', 'root_dir']:
                if param not in deficit_params:
                    raise Exception(f"Error: Blur deficit params must contain {param}")

            self.start_epoch = self.deficit_params['start_epoch']
            self.end_epoch = self.deficit_params['end_epoch']
            self.subset_size = self.deficit_params['subset_size']
            self.root_dir = self.deficit_params['root_dir']

            self.transform_train = v2.Compose([
                v2.RandomCrop(32, padding=4),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])

            self.transform_test = v2.Compose([
                #v2.ToTensor(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])

            #full_trainset = torchvision.datasets.CIFAR10(root=self.root_dir, train=True, download=True, transform=self.transform_train)
            #self.testset = torchvision.datasets.CIFAR10(root=self.root_dir, train=False, download=True, transform=self.transform_test)

            #self.trainset, _ = random_split(full_trainset, [self.subset_size, 1-self.subset_size])

            #print(f'length of trainset: {len(self.trainset)}')

            self.trainset = torchvision.datasets.CIFAR10(root=self.root_dir, train=True, download=True, transform=self.transform_train)
            self.testset = torchvision.datasets.CIFAR10(root=self.root_dir, train=False, download=True, transform=self.transform_test)

            self.sampler = RandomSampler(deficit_params['start_epoch'], deficit_params['end_epoch'],
                                        deficit_params['subset_size'], len(self.trainset))


    def update_deficit(self, epoch):
        self.sampler.set_epoch(epoch)

    def Apply_To_Experiment(self, exp):
        exp.trainloader_params['dataset'] = self.trainset
        exp.testloader_params['dataset'] = self.testset
        exp.trainloader_params['sampler'] = self.sampler

        exp.trainloader = DataLoader( **(exp.trainloader_params) )
        exp.testloader = DataLoader( **(exp.testloader_params) )


if __name__ == '__main__':
    dps =  {'start_epoch':0,
            'end_epoch':1,
            'subset_size':0.6,
            'root_dir':'./data',
            'dataset':'CIFAR10'
            }

    d = CIFAR10SubsetDeficit(dps)