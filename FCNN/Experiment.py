import torch
import torch.nn.functional as F

#class variables are going to be
# Model stuff:

#- pytorch.nn
#- optimizer
#-- optimizer parameters
#- criterion
#- dataset

#Deficit 
#- type


# General training stuff

#- device
#- epochs to train after deficit


class Model():
    def __init__(self, nn_class, nn_params, optimizer_class, optimizer_params, criterion_class, dataset, scheduler=None):
        self.nn = nn_class(**nn_params)

        #add the models initialized weights to the optimizer parameter list
        optimizer_params['params'] = self.nn.parameters()
        self.optimizer = optimizer_class(**optimizer_params)

        self.criterion = criterion_class()

        self.dataset = dataset


class Experiment():


    def __init__(self, num):
        self.num = num


        pass

    
    #def train_one_epoch(model, loader, optimizer, device):
    def train_one_epoch(self):
        model = self.model
        loader = self.train_loader
        optimizer = self.optimizer
        device = self.device
        
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




    def evaluate(self):
        model = self.model
        loader = self.test_loader
        device = self.device


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




    #def train_model(model, num_epochs, train_loader, test_loader, optimizer, scheduler, device):
    def train_model(self):

        num_epochs = self.num_epochs
        scheduler = self.scheduler

        # Store statistics in a list
        train_losses, train_accs = [], []
        test_losses, test_accs = [], []

        # Epochs loop
        for epoch in range(num_epochs):

            # Train
            #train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
            train_loss, train_acc = self.train_one_epoch()

            # Step the LR scheduler
            scheduler.step()

            # Evaluate
            #val_loss, val_acc = evaluate(model, test_loader, device)
            val_loss, val_acc = self.evaluate()
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
        return train_losses, train_accs, test_losses, test_accs

if __name__ == '__main__':
    print('hi')
    