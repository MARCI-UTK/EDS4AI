
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BlurTransform(nn.Module):
    def __init__(self, start_epoch, end_epoch, transform_size=32):
        super(BlurTransform, self).__init__()
        self.start = start_epoch
        self.end = end_epoch
        self.current_epoch = 0
        self.transform_size = transform_size

        self.layers = nn.Sequential(
            nn.MaxPool2d(4)
        )

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self,x):
        if (self.current_epoch >= self.start) and (self.current_epoch < self.end) :
            x = self.layers(x)
            #print(f"x shape {x.shape}")
            #print('transforming')
            #print(x.shape)
            x = torch.unsqueeze(x, 0)
            x = F.interpolate(x, size=(self.transform_size, self.transform_size), mode='nearest')
            x = x.squeeze(0)
            return x
        else :
            return x