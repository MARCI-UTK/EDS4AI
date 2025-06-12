import importlib

def get_class(module_name, class_name):
    module = importlib.import_module(module_name)

    if hasattr(module, class_name) :
        imported_class = getattr(module, class_name)
        return imported_class
    else :
        print(f'{class_name} class not present in {str(module)}')
        exit()