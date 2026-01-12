from exp_driver.experiment import Experiment
from implementations.deficits.BlurSubsetDeficit import BlurSubsetDeficit



exp_params = {
    "device" : "cuda:0",

    "duration" : "250",

    "deficit_params" : {
        "dataset" : "CIFAR10",
        "start_epoch" : 0,
        "end_epoch" : 1,
        "root_dir" : "../data",
        "subset_size" : 0.2,
    },
}



if __name__ == "__main__":
    experiment = Experiment(exp_params=exp_params)

    blur_def = BlurSubsetDeficit( **exp_params["deficit_params"] )

    experiment.add_deficit(blur_def)

    trainloader = experiment.trainloader

    for idx, (data, labels) in enumerate(trainloader):
        print(idx)

    print("here")