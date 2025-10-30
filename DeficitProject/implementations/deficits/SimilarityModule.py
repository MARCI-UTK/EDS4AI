
import torch
import torch.utils.data
import numpy as np
import torchvision

from torch.utils.data import DataLoader
from torchvision.transforms import v2

from exp_driver.deficit import Deficit

class SimilaritySampler(torch.utils.data.Sampler):
    def __init__(self, start_epoch, end_epoch, subset_percentage, quantiles, type='similarity', shuffle=True):
        self.quantiles = quantiles
        self.current_epoch = 0
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.type = type
        self.shuffle = shuffle
        self.all_indices = np.arange(0, quantiles.shape[0])

        if self.type == 'similarity' :
            mask = quantiles <= subset_percentage
        elif self.type == 'disimilarity':
            mask = quantiles >= 1 - subset_percentage

        self.subset = self.all_indices[mask]
        #print(f'subset: {self.subset}')

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __iter__(self):
        if self.current_epoch >= self.start_epoch and self.current_epoch < self.end_epoch:
            indices = self.subset
        else :
            indices = self.all_indices.copy()

        if self.shuffle :
            indices = indices[torch.randperm(indices.shape[0])]
        

        return iter(indices)


class SimilarityTypeDeficit(Deficit):
    def __init__(self, start_epoch, end_epoch, subset_size, quantiles, data, 
                 root_dir='../data', type='similarity', shuffle=True,):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.subset_size = subset_size
        self.quantiles = quantiles
        self.type = type
        self.shuffle = shuffle
        self.data_location = root_dir
        self.deficit_params = locals()

        if data == "CIFAR10":


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

            self.trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=self.transform_train)
            self.testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=self.transform_test)

        else :
            raise Exception(f"Dataset {data} not implemented")

        self.sampler = SimilaritySampler(subset_percentage=self.subset_size, quantiles=self.quantiles, start_epoch=self.start_epoch,
                                            end_epoch=end_epoch, type=self.type, shuffle=self.shuffle)

    
    def update_deficit(self, epoch):
        self.sampler.set_epoch(epoch)

    def Apply_To_Experiment(self, exp):
        exp.trainloader_params['dataset'] = self.trainset
        exp.testloader_params['dataset'] = self.testset
        exp.trainloader_params['sampler'] = self.sampler
        
        exp.trainloader = DataLoader( **(exp.trainloader_params) )
        exp.testloader = DataLoader( **(exp.testloader_params) )
        return super().Apply_To_Experiment(exp)