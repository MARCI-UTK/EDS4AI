import sys
import torch
import numpy as np

sys.path.insert(0, '../DeficitProject')

from exp_driver.experiment import Experiment
from exp_driver.deficit import Deficit
from exp_driver.model import Model
import importlib

SEED = 42
NUM_EPOCHS = 20

exp1 = {
    'experiment_module' : 'Experiment',
    'experiment_name' : 'Experiment',

    'device' : 'cuda:2',

    'trainloader_params': { 'batch_size' : 128, 'shuffle' : True, 'num_workers' : 2},
    'testloader_params': { 'batch_size' : 128, 'num_workers' : 1},

    'nn_module' : "implementations.models.AllCNN_WBN",
    'nn_name' : 'AllCNN_WBN',
    'nn_params' : {'num_classes' : 10, 'hidden_dropout_prob' : 0.3, 'input_dropout_prob' : 0},

    'optimizer_name' : 'SGD',
    'optimizer_params' : {'lr' : 0.05, 'weight_decay':0.001, 'momentum' : 0.9},

    'scheduler_name' : 'StepLR',
    'scheduler_params' : {'step_size':1, 'gamma':0.97},

    'criterion_name' : 'CrossEntropyLoss',

    'deficit_module' : 'implementations.deficits.BlurModule',
    'deficit_name' : 'BlurDeficit',
    # start_epoch == end_epoch == 0 means blur is never active
    'deficit_params' : {'start_epoch': 0, 'end_epoch': 0, 'root_dir': '../DeficitProject/data', 'dataset': 'CIFAR10'},

    'num_epochs' : NUM_EPOCHS,
    'output_dir' : 'debug_output',
    'save_epochs' : [],
}

def get_class(module_name, class_name):
    module = importlib.import_module(module_name)
    if hasattr(module, class_name):
        return getattr(module, class_name)
    else:
        print(f'{class_name} class not present in {str(module)}')
        exit()


if __name__ == '__main__':
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment = Experiment(exp1)

    nn_class = get_class(exp1['nn_module'], exp1['nn_name'])
    opt = get_class('torch.optim', exp1['optimizer_name'])
    scheduler_class = get_class('torch.optim.lr_scheduler', exp1['scheduler_name'])
    criterion_class = get_class('torch.nn', exp1['criterion_name'])

    model_wrapper = Model(
        nn_class=nn_class,
        nn_params=exp1['nn_params'],
        optimizer_class=opt,
        optimizer_params=exp1['optimizer_params'],
        criterion_class=criterion_class,
        trainset=None,
        testset=None,
        scheduler_class=scheduler_class,
        scheduler_params=exp1['scheduler_params'],
    )

    deficit_class = get_class(exp1['deficit_module'], exp1['deficit_name'])
    deficit = deficit_class(exp1['deficit_params'])

    experiment.add_model(model_wrapper=model_wrapper)
    experiment.add_deficit(deficit=deficit)
    experiment.train_model(verbose=True)
