
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import random

import optuna
from optuna.trial import TrialState

from utils.get_class import get_class

model_module = 'implementations.models.AllCNN_WBN'
model_name = 'AllCNN_WBN'

model_class = get_class(module_name=model_module, class_name=model_name)


def objective(trial):
    torch.manual_seed(0)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # pad 4 pixels on each side, then random crop 32x32
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Normalize to zero mean & unit variance using the
        # per-channel mean & std of CIFAR-10
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    # Transformations for test: just convert to tensor & normalize.
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    # Download & create datasets
    full_dataset_train = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform_train)
                                            
    full_dataset_val = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform_val)
                
    length_dataset = len(full_dataset_train)

    random_indices = torch.randperm(length_dataset)
    train_indices = random_indices[:int(length_dataset*0.8)]
    val_indices = random_indices[int(length_dataset*0.8):]
    
    train_dataset = Subset(full_dataset_train, train_indices)
    val_dataset = Subset(full_dataset_val, val_indices)

    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)

    valloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)



    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    #print(f"Using device: {device}")

    hidden_prob = trial.suggest_float("hidden_dropout_probability", 0, 0.5)
    input_prob = trial.suggest_float("input_dropout_probability", 0, 0.5)
    #input_prob = 0

    model = model_class(num_classes=10, hidden_dropout_prob=hidden_prob, input_dropout_prob=input_prob).to(device)
    criterion = nn.CrossEntropyLoss()

    #optimizer_name = trial.suggest_categorical("optimzer", ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay_rate", 1e-5, 1e-1, log=True)
    #gamma = trial.suggest_float("gamma_rate", 5e-2, 5e-1, log=True)


    #optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=gamma)


    # Training parameters
    num_epochs = 250

    # Store loss and accuracy
    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []


    #trainloader, testloader = get_loaders()

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        #scheduler.step()

        train_loss = running_loss / len(trainloader.dataset)
        train_acc = 100 * correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # Evaluate on test set
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss = test_loss / len(valloader.dataset)
        test_acc = 100 * correct / total
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        
        trial.report(test_acc, epoch)

        if trial.should_prune():
            print(f'Trial {trial.number} pruned')
            raise optuna.exceptions.TrialPruned()

        #print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    #print("Training Finished!")
    print(f'Trial {trial.number} completed, accuracy: {test_acc}')
    return test_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    #study.optimize(objective, n_trials=100, timeout=(60*60*24), n_jobs=-1)
    study.optimize(objective, n_trials=100, timeout=(60*60*24))

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))