import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
#from torchinfo import summary

# If running in a notebook, enable inline plots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using device:", device)

# set random seed
torch.manual_seed(45)

class AllCNN_WBN(nn.Module):
    def __init__(self, num_classes=10, hidden_dropout_prob=0.0, input_dropout_prob=0.0):
        super(AllCNN_WBN, self).__init__()
        # Note: the exact filter counts in the paper are 96 and 192.
        # Here we'll follow that fairly closely.

        self.num_classes = num_classes

        # Block 1
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, padding=1, stride=2)  # downsample
        self.bn2d_3 = nn.BatchNorm2d(96)

        # Block 2
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.bn2d_4 = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn2d_5 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, padding=1, stride=2)  # downsample
        self.bn2d_6 = nn.BatchNorm2d(192)

        # Block 3
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn2d_7 = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1, padding=0)
        self.bn2d_8 = nn.BatchNorm2d(192)
        self.conv9 = nn.Conv2d(192, num_classes, kernel_size=1, padding=0)
        self.bn2d_9 = nn.BatchNorm2d(num_classes)

        self.dropout = nn.Dropout(p=hidden_dropout_prob)
        self.input_dropout = nn.Dropout(p=input_dropout_prob)

    def forward(self, x):

        #input dropout
        x = self.input_dropout(x)

        # Block 1
        x = self.dropout(F.relu(self.bn2d_1(self.conv1(x))))
        x = self.dropout(F.relu(self.bn2d_2(self.conv2(x))))
        x = self.dropout(F.relu(self.bn2d_3(self.conv3(x))))  # output shape ~ [batch, 96, 16, 16]

        # Block 2
        x = self.dropout(F.relu(self.bn2d_4(self.conv4(x))))
        x = self.dropout(F.relu(self.bn2d_5(self.conv5(x))))
        x = self.dropout(F.relu(self.bn2d_6(self.conv6(x))))  # output shape ~ [batch, 192, 8, 8]

        # Block 3
        x = self.dropout(F.relu(self.bn2d_7(self.conv7(x))))
        x = self.dropout(F.relu(self.bn2d_8(self.conv8(x))))
        x = self.dropout(F.relu(self.bn2d_9(self.conv9(x))))  # output shape ~ [batch, 10, 8, 8]

        # Global average pooling to get 1x1, then flatten
        #  => shape [batch, 10, 1, 1]
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(-1, self.num_classes)  # shape [batch, 10]

        return x

def init_weights_xavier(m):
    """
    Custom initialization: applies Xavier initialization to Conv2D and Linear layers.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def train_model(model, num_epochs, train_loader, test_loader, optimizer, scheduler, device):
  """
  Train the model on the given data loader for a given number of epochs.
  Args:
      model: The model to train.
      num_epochs: The number of epochs to train for.
      train_loader: The data loader to use for training.
      test_loader: The data loader to use for testing.
      optimizer: The optimizer to use [Adam,SGD].
      scheduler: The learning rate scheduler to use.
      device: The device to use [GPU/CPU].
  Returns:
      model: The trained model.
      train_losses: A list of training losses for each epoch.
      train_accs: A list of training accuracies for each epoch.
      test_losses: A list of test losses for each epoch.
      test_accs: A list of test accuracies for each epoch.
  """

  # Store statistics in a list
  train_losses, train_accs = [], []
  test_losses, test_accs = [], []

  # Epochs loop
  for epoch in range(num_epochs):

    # Train
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)

    # Step the LR scheduler
    scheduler.step()

    # Evaluate
    val_loss, val_acc = evaluate(model, test_loader, device)

    # Save stats
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(val_loss)
    test_accs.append(val_acc)
    last_lr = scheduler.get_last_lr()[0]

    # Output progress
    print(f"Epoch [{epoch+1}/{num_epochs}]: "
          f" LR: {last_lr:.8f} "
          f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
          f" Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%" )

  #return
  return model, train_losses, train_accs, test_losses, test_accs


if __name__ == "__main__":
    # Transformations for training: random crop, random horizontal flip,
    # then convert to tensor & normalize.
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
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    # Download & create datasets
    train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform_train)

    test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transform_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset,
                            batch_size=128,
                            shuffle=True,
                            num_workers=2)  # adjust num_workers to your CPU

    test_loader = DataLoader(test_dataset,
                            batch_size=128,
                            shuffle=False,
                            num_workers=2)

    # Hyperparameters
    lr = 0.05            # initial learning rate
    momentum = 0.9
    weight_decay = 1e-3
    num_epochs = 200      # you may go higher (100, 200, or 350 as in the paper)
    hidden_drop = 0.3
    input_drop = 0.0
    milestones = [50, 100, 150]
    gamma = 0.1

    # Model
    # Instantiate the model, apply Xavier init, move to device
    model = AllCNN_WBN(num_classes=10, hidden_dropout_prob=hidden_drop, input_dropout_prob=input_drop)
    model.apply(init_weights_xavier)  # apply our init function to each layer
    model = model.to(device)
    #summary(model)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(),
                        lr=lr,                      momentum=momentum,
                        weight_decay=weight_decay)

    # Optionally define a learning rate schedule:
    # For example, reduce LR by 10 at epochs [25, 40]
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    #                                            milestones=[200, 250, 300],
    #                                            gamma=0.1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    #                                            milestones=milestones,
    #                                            gamma=gamma)
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.97)

    m, train_losses, train_accs, test_losses, test_accs = train_model(model, num_epochs, train_loader, test_loader, optimizer, scheduler, device)

    print("\n## Experiment #:")
    #Print Hyperparameters
    print(f"\n### Hyperparameters:")
    print(f"- hidden_drop: {hidden_drop}")
    print(f"- input_drop: {input_drop}")
    print("\nOptimizer:")
    print(f"- optimizer: {type(optimizer)}")
    print(f"- Initial lr: {lr}")
    print(f"- momentum: {momentum}")
    print(f"- weight_decay: {weight_decay}")
    print(f"- num_epochs: {num_epochs}")
    print("\nScheduler:")
    print(f"- type: {type(scheduler)}")
    print(f"- milestones: {milestones}")
    print(f"- gamma: {gamma}")

    # Print final accuracy
    print("\n### Results:")
    print(f"- Final Training Accuracy: {train_accs[-1]:.2f}%")
    print(f"- Final Test Accuracy: {test_accs[-1]:.2f}%")

    # Plot accuracy curves
    plt.figure(figsize=(10,5))
    plt.title("Accuracy vs. Epoch")
    plt.plot(train_accs, label="Train")
    plt.plot(test_accs, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig("AllCNN_WBN_Acc.png")

    # Plot loss curves
    plt.figure(figsize=(10,5))
    plt.title("Loss vs. Epoch")
    plt.plot(train_losses, label="Train")
    plt.plot(test_losses, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("AllCNN_WBN_Loss.png")