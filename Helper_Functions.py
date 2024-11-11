import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import random_split
import torch.nn.functional as F

def calculate_accuracy(model, testloader, device):
    model.eval()

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

    model.train()

    return correct_predictions_total/running_samples_total



def calculate_valloss(model, criterion, valloader, device):
    model.eval()

    with torch.no_grad():
        n_batches = 0
        for i, data in enumerate(valloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            n_batches += 1

    model.train()

    return loss/n_batches


import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    checkpoint = {'epoch': epoch, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss } 
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from epoch {epoch}, loss: {loss}")
    return model, optimizer, epoch, loss
