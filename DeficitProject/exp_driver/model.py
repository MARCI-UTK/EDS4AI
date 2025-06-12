import torch

class Model():
    def __init__(self, nn_class, nn_params, optimizer_class, optimizer_params, criterion_class,
                 trainset, testset, scheduler_class=None, scheduler_params=None):

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

            #this scheduler does nothing
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=1)
            #self.scheduler_name = self.

        else :
            scheduler_params['optimizer'] = self.optimizer
            self.scheduler_params = scheduler_params
            self.scheduler_params['optimizer'] = self.optimizer
            self.scheduler = torch.optim.lr_scheduler.StepLR( **(self.scheduler_params) )

        self.trainset = trainset
        self.testset = testset