
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