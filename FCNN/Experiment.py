import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import string
import numpy as np
import pandas as pd
import random
import pathlib
import json
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt

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

#import json
#def safe_default(obj):
    #if type(obj) == type:
        #return (obj).__name__
    #else:
        #return str(obj)

#def serialize_experiment_params(experiment, filename):
    #obj = {}

    #for field in experiment.info_to_save :
        #obj[field] = getattr(experiment, field)
    

    #default = lambda o: safe_default(o)
    #with open(filename, 'w') as file:
        #json.dump(obj, file, default=default,indent=4)


#abstract base class for deficits, every deficit must implement the function
#get_sampler which returns an instance of a class that inherits torch.utils.data.Sampler
#from abc import ABC, classmethod
from abc import ABC, abstractmethod
class Deficit(ABC):
    def __init__(self, deficit_class, deficit_params):
        self.deficit_class = deficit_class
        self.deficit_params = deficit_params
        self.deficit = deficit_class(deficit_params)

    #@classmethod 
    @abstractmethod
    def update_deficit(self, epoch):
        pass

    @abstractmethod
    def Apply_To_Experiment(self, exp):
        pass

class Model():
    def __init__(self, nn_class, nn_params, optimizer_class, optimizer_params, criterion_class,
                 trainset, testset, scheduler_class=None):

        self.nn_class = nn_class
        self.nn_params = nn_params    
        self.nn = nn_class(**nn_params)

        #add the models initialized weights to the optimizer parameter list
        optimizer_params['params'] = self.nn.parameters()

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.optimizer = optimizer_class(**optimizer_params)

        self.criterion_class = criterion_class
        self.criterion = criterion_class()

        if scheduler_class == None: 
            self.scheduler_class = None

            #this scheduler does nothing
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=1)

        self.trainset = trainset
        self.testset = testset


class Experiment():
    info_to_save = [
                    'num_epochs', 'device', 'model_name', 'model_params', 'optimizer_name', 'optimizer_params',
                    'criterion_name', 'deficit_name', 'deficit_params', 'trainloader_params', 'testloader_params',

                    ]


    def __init__(self, exp_params):
        if 'num_epochs' in exp_params:
            self.num_epochs = exp_params['num_epochs']
        else :
            self.num_epochs = 200

        if 'device' in exp_params:
            self.device = exp_params['device']
        else :
            self.device = 'cpu'

        if 'trainloader_params' in exp_params:
            self.trainloader_params = exp_params['trainloader_params']
        else :
            self.trainloader_params = {'batch_size' : 128, 'shuffle' : True}

        if 'testloader_params' in exp_params:
            self.testloader_params = exp_params['testloader_params']
        else :
            self.testloader_params = {'batch_size' : 128, 'shuffle' : False}


        if 'output_dir' in exp_params:
            self.output_dir = exp_params['output_dir']
        else :
            self.output_dir = 'output'


    def add_model(self, model_wrapper : Model):
        self.model = model_wrapper.nn
        self.model_params = model_wrapper.nn_params
        self.model_class = model_wrapper.nn_class
        self.model_name = self.model_class.__name__
        
        self.optimizer = model_wrapper.optimizer
        self.optimizer_params = model_wrapper.optimizer_params
        self.optimizer_class = model_wrapper.optimizer_class
        self.optimizer_name = self.optimizer_class.__name__
        
        self.criterion = model_wrapper.criterion
        self.criterion_class = model_wrapper.criterion_class
        self.criterion_name = self.criterion_class.__name__

        self.trainset = model_wrapper.trainset
        self.testset = model_wrapper.testset

        self.scheduler = model_wrapper.scheduler



    #here you pass an instance of the deficit class
    def add_deficit(self, deficit=None):
        #we can't create the the dataloaders until we know what the deficit is
        # this is because we need to know if there is a sampler

        # blank deficit doesn't need a sampler
        if deficit == None:
            self.trainloader_params['dataset'] = self.trainset
            self.testloader_params['dataset'] = self.testset

            self.trainloader = DataLoader( **(self.trainloader_params) )
            self.testloader = DataLoader( **(self.testloader_params) )


            self.deficit = None
            self.deficit_duration = 0
        else :
            self.deficit = deficit
            self.deficit_params = deficit.deficit_params
            self.deficit_name = type(self.deficit).__name__
            #THis  will set the trainloader
            deficit.Apply_To_Experiment(self)


    
    #def train_one_epoch(model, loader, optimizer, device):
    def train_one_epoch(self):
        model = self.model
        loader = self.trainloader
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
        loader = self.testloader
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
        #self.exp_id = ''.join(random.choices(string.ascii_letters + string.digits, k = 8))
        #self.info_to_save.append('exp_id') 

        self.model.to(self.device)

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


        self.exp_id = ''.join(random.choices(string.ascii_letters + string.digits, k = 8))
        self.info_to_save.append('exp_id') 

        dir = self.output_dir + '/data/' + self.exp_id + '/'

        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(train_losses, columns=['train_loss']) 
        df.to_csv(dir + 'train_losses.csv', index=False)

        df = pd.DataFrame(train_accs, columns=['train_acc']) 
        df.to_csv(dir + 'train_accs.csv', index=False)

        df = pd.DataFrame(test_losses, columns=['test_loss']) 
        df.to_csv(dir + 'test_losses.csv', index=False)

        df = pd.DataFrame(test_accs, columns=['test_acc']) 
        df.to_csv(dir + 'test_accs.csv', index=False)

        #write the json config file now 
        # the config will be saved twice, once with the name config.json in the directory
        # in data corresponding to exp_id, the second time it is saved to configs with 
        # the file name exp_id.json

        self.serialize_experiment_params(dir + "config.json")
    
        dir2 = self.output_dir + "/configs/"
        pathlib.Path(dir2).mkdir(parents=True, exist_ok=True)
        self.serialize_experiment_params(dir2 + self.exp_id + ".json")
        


        return train_losses, train_accs, test_losses, test_accs


    def serialize_experiment_params(self, filename):

        def safe_default(obj):
            if type(obj) == type:
                return (obj).__name__
            else:
                return str(obj)

        obj = {}

        for field in self.info_to_save :
            obj[field] = getattr(self, field)

        default = lambda o: safe_default(o)
        with open(filename, 'w') as file:
            json.dump(obj, file, default=default,indent=4)



def get_data(dir, exp_id):
    if not os.path.isdir(dir):
        raise FileNotFoundError("Directory does not exist: {dir}")

    fname = dir + '/data/' + exp_id + '/'

    df = pd.read_csv(fname + 'train_losses.csv')
    train_losses = df['train_loss'].tolist()

    df = pd.read_csv(fname + 'train_accs.csv')
    train_accs = df['train_acc'].tolist()

    df = pd.read_csv(fname + 'test_losses.csv')
    test_losses = df['test_loss'].tolist()

    df = pd.read_csv(fname + 'test_accs.csv')
    test_accs = df['test_acc'].tolist()

    return train_losses, train_accs, test_losses, test_accs


# taken from chatgpt
def is_subdict(small, big):
    if not isinstance(small, dict) or not isinstance(big, dict):
        return False
    for key, value in small.items():
        if key not in big:
            return False
        if isinstance(value, dict):
            if not isinstance(big[key], dict):
                return False
            if not is_subdict(value, big[key]):
                return False
        else:
            if big[key] != value:
                return False
    return True


# takes a list of directoroies and looks through each of those to 
# see if there exists an experiment that matches the params passed
# returns a list of exp_ids
def match_experiments(directories, params):
    exp_ids = []
    for dir in directories:

        for file in pathlib.Path(dir + '/configs').iterdir():
            if file.is_file():
                fp = file.open()
                config = json.load(fp)
                fp.close()

                if is_subdict(params, config):
                    exp_ids.append(config['exp_id'])

    return exp_ids


def plot_exp(dir, exp_id):
    tr_l, tr_a, te_l, te_a = get_data(dir, exp_id=exp_id)

    epoch_list = np.arange(1, len(tr_l) + 1)

    dic = {'losses':tr_l, 'epoch':epoch_list}
    df = pd.DataFrame(data=dic)
    #df = df[150:]

    p = sns.lineplot(data=df, x=df.index, y='losses')
    p.set_xlabel("Epoch")
    p.set_ylabel("Avg Cross Entropy Loss")
    p.set_title("Training Loss")

    plt.savefig('plot.png')




small = {
    "a": 1,
    "b": {
        "x": 10,
        'sub': {'one': 1}
    }
}

big = {
    "a": 1,
    "b": {
        "x": 10,
        "y": 20,
        'sub': {'on': 2, 'two': 2}
    },
    "c": 999
}


if __name__ == '__main__':
    #print( is_subdict(small=small, big = big))
    print('hi')

    plot_exp('output', 'jmfCGLMA')
    