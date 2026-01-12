import numpy as np

from exp_driver.experiment import Experiment
from exp_driver.deficit import Deficit
from exp_driver.model import Model
import importlib

exp1 = {
    'experiment_module' : 'Experiment',
    'experiment_name' : 'Experiment',

    'device' : 'cuda:2',

    'trainloader_params': { 'batch_size' : 128, 'num_workers' : 2},
    'testloader_params': { 'batch_size' : 128, 'num_workers' : 1},

    'nn_module' : "implementations.models.AllCNN_WBN",
    'nn_name' : 'AllCNN_WBN',
    'nn_params' : {'num_classes' : 10, 'hidden_dropout_prob' : 0.3, 'input_dropout_prob' : 0},

    'optimizer_name' : 'Adam',
    'optimizer_params' : {'lr' : 0.001},

    'criterion_name' : 'CrossEntropyLoss',

    #'deficit_module' : 'implementations.deficits.cifar10_subset_deficit',
    'deficit_module' : 'implementations.deficits.BlurModule',
    'deficit_name' : 'BlurDeficit',
    'deficit_params' : {'start_epoch':0, 'root_dir':'../data', 'dataset':'CIFAR10'},
    #'deficit_params' : {'start_epoch':0},

}
    
def get_class(module_name, class_name):
    module = importlib.import_module(module_name)

    if hasattr(module, class_name) :
        imported_class = getattr(module, class_name)
        return imported_class
    else :
        print(f'{class_name} class not present in {str(module)}')
        exit()


durations = [300, 450]
#durations = [1]
#durations = [320, 360, 400, 440, 480, 520]
post_duration = 250
#post_duration = 1

if __name__ == '__main__' :

    for deficit_duration in durations:
        exp1['deficit_params']['end_epoch'] = deficit_duration 
        exp1['num_epochs'] = deficit_duration + post_duration
        exp1['output_dir'] = 'studies/blur/blur_weights'

        #exp1['save_epochs'] = [item for item in [deficit_duration-1, deficit_duration-1+post_duration] if item >= 0]
        exp1['save_epochs'] = list(range(deficit_duration + post_duration))


        trainset, testset = None, None

        #quantiles = np.load('studies/subset_size/CIFAR10_train_quantiles.npy')
        #exp1['deficit_params']['quantiles'] = quantiles

        experiment = Experiment(exp1)

        nn_class = get_class(exp1['nn_module'], exp1['nn_name'])
        nn_params = exp1['nn_params']

        opt = get_class('torch.optim', exp1['optimizer_name'])
        opt_params = exp1['optimizer_params']


        criterion_class = get_class('torch.nn', exp1['criterion_name'])

        model_wrapper = Model(nn_class=nn_class, nn_params=nn_params, optimizer_class=opt, optimizer_params=opt_params,
                                        criterion_class=criterion_class, trainset=trainset, testset=testset,
                                        )


        deficit_class = get_class(exp1['deficit_module'], exp1['deficit_name'])
        deficit_params = exp1['deficit_params']
        deficit = deficit_class(deficit_params)

        experiment.add_model(model_wrapper=model_wrapper)


        experiment.add_deficit(deficit=deficit)

        experiment.train_model(verbose=True)