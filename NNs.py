import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import random_split
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):

    
    def __init__(self):
        super().__init__()

        
        self.InputDropout = nn.Dropout(p=0.2)

        self.ConvBlock1 = torch.nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            #nn.Dropout(0.2),
        )

        self.ConvBlock2 =  torch.nn.Sequential(
            nn.Conv2d(96, 96, 3,padding=1),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            #nn.Dropout(0.3),
        )

        #self.PoolBlock1 = torch.nn.Sequential(
            #nn.Conv2d(96, 96, 3, stride=2),
            #nn.BatchNorm2d(num_features=96),
            #nn.ReLU(),
            #nn.Dropout()
        #)

        self.PoolBlock1 = torch.nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.ConvBlock3 = torch.nn.Sequential(
            nn.Conv2d(96, 192, 3, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            #nn.Dropout(0.3),
        )

        self.ConvBlock4 = torch.nn.Sequential(
            nn.Conv2d(192, 192, 3, padding=1, ),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
        )


        #self.PoolBlock2 = torch.nn.Sequential(
            #nn.Conv2d(192, 192, 3, stride=2),
            #nn.BatchNorm2d(num_features=192),
            #nn.ReLU(),
            #nn.Dropout()
        #)
        
        self.PoolBlock2 = torch.nn.Sequential(
            nn.Conv2d(192, 192, 3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        #self.ConvBlock5 = torch.nn.Sequential(
            #nn.Conv2d(192, 192, 3, padding=1),
            #nn.BatchNorm2d(num_features=192),
            #nn.ReLU(),
            ##nn.Dropout(),
        #)

        self.ConvBlock5 = torch.nn.Sequential(
            nn.Conv2d(192, 192, 3),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            #nn.Dropout(),
        )

        self.LinLayer1 = torch.nn.Sequential(
            nn.Conv2d(192, 192, 1),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
        )

        self.LinLayer2 = torch.nn.Sequential(
            nn.Conv2d(192, 10, 1),
            nn.BatchNorm2d(num_features=10),
            nn.ReLU(),
        )

        self.FinalPool = nn.AvgPool2d(6)


    def forward(self, x):
        #x = self.ConvLayers(x)
        #x = torch.squeeze(x)

        # return x

        x = self.InputDropout(x)

        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        
        x = self.PoolBlock1(x)

        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)

        x = self.PoolBlock2(x)

        x = self.ConvBlock5(x)

        x = self.LinLayer1(x)
        x = self.LinLayer2(x)

        x = self.FinalPool(x)

        x = torch.squeeze(x)

        return x


def calculate_accuracy(model, testloader, device):
    with torch.no_grad():

        running_samples_total = 0
        correct_predictions_total = 0
            
        for indx, data in enumerate(testloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            prediction = model(inputs)
            _, tensor_index = torch.max(prediction, dim=1)

            
            correct_predictions_total += (len(labels[labels==tensor_index]))
            running_samples_total += prediction.shape[0]

        return correct_predictions_total/running_samples_total


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x



#def init_weights(m, generator=np.random.default_rng(0)):
    #bias_val = generator.uniform(-0.05, 0.05)
    #print(bias_val)

    #if isinstance(m, nn.Linear):
        #torch.nn.init.xavier_uniform_(m.weight)
        #m.bias.data.fill_(bias_val)
    
def init_weights(m, xavier_scale=1):
    bias_val = torch.FloatTensor(1).uniform_(-0.05, 0.05)[0]

    #print(bias_val)

    if isinstance(m, nn.Linear):
        print('here')
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(bias_val)

    if isinstance(m, nn.Conv2d):
        print(bias_val)
        #print('bruh')
        torch.nn.init.xavier_uniform_(m.weight, xavier_scale)
        m.bias.data.fill_(bias_val)
        print(m.weight)


class Blur(nn.Module):
    def __init__(self):
        super(Blur, self).__init__()
        self.layers = torch.nn.Sequential(
            nn.MaxPool2d(4)
        )

    def forward(self,x):
       x = self.layers(x)
       #print(x.shape)
       x = F.interpolate(x, size=(32, 32), mode='nearest')
       return x