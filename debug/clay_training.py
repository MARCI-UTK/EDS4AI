import sys
sys.path.insert(0, '../DeficitProject')

from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
import torch.nn as nn
from BlurTransform import BlurTransform
import os
import numpy as np

import time

# GRPOLoss not available; stub included in case --loss_function GRPO is passed
class GRPOLoss(nn.Module):
    def __init__(self, beta=0):
        raise NotImplementedError("GRPOLoss is not available in this debug copy")

from implementations.models.AllCNN_WBN import AllCNN_WBN
from datetime import datetime
import json

import argparse

SEED = 42

class PreprocessedCIFAR10(Dataset):
    def __init__(self, path, augment=None):
        self.data, self.targets = torch.load(path)
        self.augment = augment

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        if self.augment:
            from torchvision.transforms.functional import to_pil_image
            img = self.augment(img)
        return img, label


parser = argparse.ArgumentParser(description="Training script")

parser.add_argument("--start_epoch", type=int, default=0, help="Start Epoch for Deficit")
parser.add_argument("--device", type=int, default=0, help="Base model file or the name of model")
parser.add_argument("--end_epoch", type=int, default=10, help="End Epoch for Deficit")
parser.add_argument("--extra_epoch", type=int, default=250, help="End Epoch for Deficit")
parser.add_argument("--model_name", type=str, default='', help="Base model file or the name of model")
parser.add_argument("--model_preload", type=str, default='', help="Base model file or the name of model")
parser.add_argument("--loss_function", type=str, default='', help="Base model file or the name of model")
parser.add_argument("--optimizer", type=str, default='', help="Optimizer option. SGD or Adam")
parser.add_argument("--use_scheduler", action="store_true")
parser.add_argument("--lr", type=float, default=.05, help="Learning Rate")

args = parser.parse_args()

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def test_accuracy(model, val_loader, criterion):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    return correct / total, total_loss / len(val_loader)


model = AllCNN_WBN()
if '.pth' in args.model_preload:
    model_file = args.model_preload
    model.load_state_dict(torch.load(model_file))

model_name = args.model_name

gradients = {}

def save_gradient(name):
    def hook(grad):
        gradients[name] = grad.detach().cpu().abs().mean().item()
    return hook

for name, param in model.named_parameters():
    if param.requires_grad:
        param.register_hook(save_gradient(name))

start_epoch = args.start_epoch
end_epoch = args.end_epoch

blur_transform = BlurTransform(args.start_epoch, args.end_epoch)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.247, 0.243, 0.261)
    ),
])

transform_test = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.247, 0.243, 0.261)
    ),
])

from torchvision.datasets import CIFAR10

train_dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

val_dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

clean_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, num_workers=1)

if args.end_epoch > 0:
    transform_preprocessed = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ])
    deficit_dataset = PreprocessedCIFAR10(path="preprocessed/cifar10_train.pt", augment=transform_preprocessed)
    deficit_loader = DataLoader(deficit_dataset, batch_size=128, shuffle=True)


device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

model = model.to(device)

epochs = args.end_epoch + args.extra_epoch

correct = 0
total = 0
if args.loss_function == 'GRPO':
    criterion = GRPOLoss(beta=0)
else:
    criterion = nn.CrossEntropyLoss()

if args.optimizer == 'SGD':
    print("SGD")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=.001)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=.001)

if args.use_scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.97)

start = time.time()
test_acc, test_l = test_accuracy(model, val_loader, criterion)
end = time.time()
print(f'Test Accuracy: {100 * test_acc:.2f}%')
metrics = []
metrics.append({"training_loss":0,
                "training_accuracy":0,
                "test_loss": test_l,
                "test_accuracy": test_acc,
                "train_time":end - start})

gradients_per_epoch = []

if args.end_epoch > 0:
    train_loader = deficit_loader
else:
    train_loader = clean_loader

for epoch in range(epochs):
    if epoch == args.end_epoch and args.end_epoch > 0:
        del deficit_loader
        torch.cuda.empty_cache()
        train_loader = clean_loader
    total_loss = 0
    start_time = time.time()
    model.train()
    total = 0
    correct = 0
    start_epoch = time.time()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        output = model(images)

        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if args.use_scheduler:
        scheduler.step()

    end_epoch = time.time()

    test_acc, test_loss = test_accuracy(model, val_loader, criterion)
    print("Epoch: {}, Loss: {}, Time: {:.3f}, Test Acc: {:.3f}".format(epoch, total_loss/len(train_loader), end_epoch-start_epoch, test_acc))
    print(f'LR: {optimizer.param_groups[0]["lr"]}')
    print(f'use_scheduler: {args.use_scheduler}')

    metrics.append({"training_loss":total_loss/len(train_loader),
                    "training_accuracy":correct/total,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "train_time":end_epoch-start_epoch})
    os.makedirs("models_study/"+model_name, exist_ok=True)
    torch.save(model.state_dict(), "models_study/{}/{}_{}_epoch_{}.pth".format(model_name, model_name, datetime.now().strftime("%Y%m%d"), epoch))
    gradients_per_epoch.append({k: float(v) for k, v in gradients.items()})

    with open("models_study/{}/{}_metrics_{}.json".format(model_name, model_name, datetime.now().strftime("%Y%m%d")), "w") as f:
        json.dump(metrics, f, indent=4)

    with open("models_study/{}/{}_gradients_by_epoch_{}.json".format(model_name, model_name, datetime.now().strftime("%Y%m%d")), "w") as f:
        json.dump(gradients_per_epoch, f, indent=4)
