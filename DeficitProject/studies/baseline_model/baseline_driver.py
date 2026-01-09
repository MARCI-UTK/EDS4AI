import numpy as np

from exp_driver.experiment import Experiment
from exp_driver.deficit import Deficit
from exp_driver.model import Model
import importlib

exp1 = {
    'experiment_module' : 'Experiment',
    'experiment_name' : 'Experiment',

    'device' : 'cuda:1',

    'trainloader_params': { 'batch_size' : 128, 'num_workers' : 2},
    'testloader_params': { 'batch_size' : 128, 'num_workers' : 2},

    'nn_module' : "implementations.models.AllCNN_WBN",
    'nn_name' : 'AllCNN_WBN',
    'nn_params' : {'num_classes' : 10, 'hidden_dropout_prob' : 0.3, 'input_dropout_prob' : 0},

    'optimizer_name' : 'SGD',
    'optimizer_params' : {'lr' : 0.05, 'weight_decay':0.001, 'momentum' : 0.9},

    'scheduler_name' : 'StepLR',
    'scheduler_params' : {'step_size':1, 'gamma':0.97},

    'criterion_name' : 'CrossEntropyLoss',

    'deficit_module' : 'implementations.deficits.cifar10_subset_deficit',
    'deficit_name' : 'CIFAR10SubsetDeficit',
    'deficit_params' : {'root_dir':'../data', 'dataset':'CIFAR10'},

}
    
def get_class(module_name, class_name):
    module = importlib.import_module(module_name)

    if hasattr(module, class_name) :
        imported_class = getattr(module, class_name)
        return imported_class
    else :
        print(f'{class_name} class not present in {str(module)}')
        exit()


window_duration = 0
subset_size = 1
#overall_duration = 520

#overall_duration = 500
overall_duration = 200

iterations = 1

if __name__ == '__main__' :

    for i in range(iterations):
        exp1['deficit_params']['start_epoch'] = 0 
        exp1['deficit_params']['end_epoch'] = 0
        exp1['deficit_params']['subset_size'] = subset_size
        exp1['num_epochs'] = overall_duration
        exp1['output_dir'] = 'studies/baseline_model/baseline'

        trainset, testset = None, None

        experiment = Experiment(exp1)

        nn_class = get_class(exp1['nn_module'], exp1['nn_name'])
        nn_params = exp1['nn_params']

        opt = get_class('torch.optim', exp1['optimizer_name'])
        opt_params = exp1['optimizer_params']

        scheuduler = get_class('torch.optim.lr_scheduler', exp1['scheduler_name'])
        scheduler_params = exp1['scheduler_params']

        criterion_class = get_class('torch.nn', exp1['criterion_name'])

        model_wrapper = Model(nn_class=nn_class, nn_params=nn_params, optimizer_class=opt, optimizer_params=opt_params,
                                        criterion_class=criterion_class, trainset=trainset, testset=testset, scheduler_class=scheuduler,
                                        scheduler_params=scheduler_params)


        deficit_class = get_class(exp1['deficit_module'], exp1['deficit_name'])
        deficit_params = exp1['deficit_params']
        deficit = deficit_class(deficit_params)

        experiment.add_model(model_wrapper=model_wrapper)


        experiment.add_deficit(deficit=deficit)

        experiment.train_model(verbose=True)
        experiment.save_model("model_state_dict.pth")