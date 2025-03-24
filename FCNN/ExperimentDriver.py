
import Experiment

exp1 = {
    'experiment_module' : 'Experiment',
    'experiment_name' : 'Experiment',

    'device' : 'cpu',

    'trainloader_params': { 'batch_size' : 128, 'shuffle' : True},
    'testloader_params': { 'batch_size' : 128, 'shuffle' : False},

    'nn_module' : "FCNN",
    'nn_name' : 'FCN_CIFAR10',
    'nn_params' : {'num_classes' : 10},

    'optimizer_name' : 'AdamW',
    'optimizer_params' : {'lr' : 0.001},

    'criterion_name' : 'CrossEntropyLoss',

    'num_epochs' : 199,

}
    

# code was taken from ai generated google results
# this was the search term: 
# "python how to create a class instance from a class name passed as a string"
def get_class(module_name, class_name):
    try :
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not get {class_name} from {module_name}. {e}")
        return None 

import json
#def is_jsonable(x):
    #try:
        #json.dump(x)
        #return True
    #except(TypeError, OverflowError):
        #return False

#def convert_to_json(z):
    #if (type(z) == dict):
        #for key in z:
            #z[key] = convert_to_json(z[key])

    #else:
        #if(is_jsonable(z)):
            #return z
        #else:
            #return str(z)
        
#def safe_default(obj):
    #if type(obj) == type:
        #return (obj).__name__
    #else:
        #return str(obj)

#def safe_serialize(obj, filename):
    #default = lambda o: safe_default(o)
    #with open(filename, 'w') as file:
        #json.dump(obj, file, default=default,indent=4)

if __name__ == '__main__' :

    from Trial import get_datasets
    trainset, testset = get_datasets()

    experiment = Experiment.Experiment(exp1)

    nn_class = get_class(exp1['nn_module'], exp1['nn_name'])
    nn_params = exp1['nn_params']

    opt = get_class('torch.optim', exp1['optimizer_name'])
    opt_params = exp1['optimizer_params']

    criterion_class = get_class('torch.nn', exp1['criterion_name'])

    model_wrapper = Experiment.Model(nn_class=nn_class, nn_params=nn_params, optimizer_class=opt, optimizer_params=opt_params,
                                    criterion_class=criterion_class, trainset=trainset, testset=testset, scheduler_class=None)


    experiment.add_model(model_wrapper=model_wrapper)

    experiment.add_deficit(deficit=None)


    field_dict = {}
    for field in experiment.info_to_save :
        #print(f'{field} --->   {getattr(experiment, field)}')
        field_dict[field] = getattr(experiment, field)


    print(type(nn_class))
    print(type(nn_class) == type)

    #print(type(field_dict['testloader_params']['dataset']))
    #import inspect
    #print(  inspect.isclass(field_dict['testloader_params']['dataset']))

    #import pprint
    #pprint.pprint(field_dict)

    obj = {"a": 1, "b": bytes()} # bytes is non-serializable by default

    Experiment.serialize_experiment_params(experiment, 'hi2.json')

    #print(safe_serialize(obj))

    #print(safe_serialize(field_dict))
    #new_dict = safe_serialize(field_dict)


    #import json
    #with open('data.json', 'w') as file:
        #json.dump(new_dict, file, indent=4)

    #trainloader = DataLoader(trainset, batch_size=128)
    #testloader = DataLoader(testset, batch_size=128, shuffle=False)

    #model = model_wrapper.nn
    #loader = trainloader 
    #optimizer = model_wrapper.optimizer
    #criterion = model_wrapper.criterion
    #device = "cpu"
    
    #model.train()
    #running_loss = 0.0
    #correct = 0
    #total = 0

    #for i in range(20):
        #for images, labels in loader:
            #images, labels = images.to(device), labels.to(device)

            ## Forward pass
            #outputs = model(images)
            #loss = criterion(outputs, labels)

            ## Backward pass
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()

            ## Statistics
            #running_loss += loss.item() * images.size(0)
            #_, predicted = outputs.max(1)
            #correct += predicted.eq(labels).sum().item()
            #total += labels.size(0)

        #epoch_loss = running_loss / total
        #epoch_acc = 100.0 * correct / total
        #print(f'{epoch_loss}, {epoch_acc}')

