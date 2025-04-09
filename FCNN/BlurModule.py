
from Experiment import Deficit
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import torchvision
from torch.utils.data import DataLoader

from Trial import get_datasets

class BlurTransform(nn.Module):
    def __init__(self, start_epoch, end_epoch):
        super(BlurTransform, self).__init__()
        self.start = start_epoch
        self.end = end_epoch
        self.current_epoch = 0

        self.layers = nn.Sequential(
            nn.MaxPool2d(4)
        )

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self,x):
        if (self.current_epoch >= self.start) and (self.current_epoch < self.end) :
            x = self.layers(x)
            #print(x.shape)
            x = F.interpolate(x, size=(32, 32), mode='nearest')
            return x
        else :
            return x


class BlurDeficit(Deficit):
    def __init__(self, deficit_params):
        #for param in ['start_epoch', 'end_epoch', 'dataset']:

        if deficit_params['dataset'] == 'CIFAR10':
            for param in ['start_epoch', 'end_epoch', 'root_dir']:
                if param not in deficit_params:
                    raise Exception(f"Error: Blur deficit params must contain {param}")

            dict = deficit_params
            start_epoch = dict['start_epoch']
            end_epoch = dict['end_epoch']
            root_dir = dict['root_dir']
            self.blur_transform = BlurTransform(start_epoch, end_epoch)

            self.transform_train = v2.Compose([
                v2.RandomCrop(32, padding=4),
                v2.RandomHorizontalFlip(),
                v2.ToTensor(),
                v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                self.blur_transform,
            ])

            self.transform_test = v2.Compose([
                v2.ToTensor(),
                v2.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])

            self.trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=self.transform_train)
            self.testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=self.transform_test)

        else :
            raise Exception(f"Dataset {deficit_params['dataset']} not implemented")

    def update_deficit(self, epoch):
        self.transform.set_epoch(epoch)

    def Apply_To_Experiment(self, exp):
        exp.trainloader_params['dataset'] = self.trainset
        exp.testloader_params['dataset'] = self.testset
        
        exp.trainloader = DataLoader( **(exp.trainloader_params) )
        exp.testloader = DataLoader( **(exp.testloader_params) )
        


