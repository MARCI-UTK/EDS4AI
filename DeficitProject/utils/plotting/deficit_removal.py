import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from exp_driver.experiment import get_data, get_config, match_experiments

#def plot_all_deficit_removal_with_size(exp_ids_list, deficit_names, title="Deficit Removal", 
                             #filename="all_deficit_removal.png"):
def plot_all_deficit_removal_with_size(end_epochs, subset_sizes, params_list, directories,
                                       line_labels, title='deficit_removal', filename='fig.png'):
                             
    
    accuracies = []
    epochs = []
    labels = []
    sizes = []

    #for epoch in end_epochs:
    for subset_size in subset_sizes:
        for i, lab in enumerate(line_labels):
            for epoch in end_epochs:
                params = params_list[i]
                params['deficit_params']['subset_size'] = subset_size
                params['deficit_params']['end_epoch'] = epoch 
                exp_id, dir = match_experiments(directories, params)[0]
                _, _, _, test_accs = get_data(dir=dir, exp_id=exp_id)
                acc = test_accs[-1]

                accuracies.append(acc)
                epochs.append(epoch)
                labels.append(lab)
                sizes.append(subset_size)
                
    df = pd.DataFrame({
        "Accuracy" : accuracies,
        "Epoch" : epochs,
        "Deficit Type" : labels,
        "Subset Size" : sizes,
    })

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df,
        x="Epoch",
        y="Accuracy",
        hue="Subset Size",
        style="Deficit Type",
        #palette="viridis",
        palette="Set1",
        linewidth=2
    )
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(title="Subset Size / Deficit Type")
    plt.tight_layout()

    plt.savefig(filename)


    #for i, deficit_name in enumerate(deficit_names):
        #accuracies = {}

        #for exp_id, dir in exp_ids_list[i]:
            #_, _, _, test_accs = get_data(dir, exp_id)
            #config = get_config(exp_id, dir)

            #end_epoch = config['deficit_params']['end_epoch']
            #acc = test_accs[-1]

            #if end_epoch not in accuracies:
                #accuracies[end_epoch] = acc
            #else:
                #print(f'Already plotted end epoch {end_epoch}')

        #x = x + list(accuracies.keys())
        #y = y + list(accuracies.values())

        #name_list = [deficit_name] * len(accuracies)
        #z = z + name_list

        ##print(f'length x: {len(x)}, length z: {len(z)}')
    #df = pd.DataFrame({'epoch':x, 'accuracy':y, "deficit_name":z})
    #s = sns.lineplot(data=df, x='epoch', y='accuracy', hue='deficit_name', marker='o')
    #s.set_title(title)
    #plt.savefig(filename)
    #return s




# CHATGPT Slop im basing my code off

if __name__ == "__main__":
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    subset_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    deficit_types = ['dissimilarity', 'randomness']
    epochs = list(range(1, 11))  # 10 epochs

    epoch = []
    accuracy = []
    deficit_type = []
    subset_size = []

    for s in subset_sizes:
        for d in deficit_types:
            for e in epochs:
                epoch.append(e)
                accuracy.append(0.6 + 0.03 * e + 0.1 * s + (0.02 if d == 'dissimilarity' else 0))
                deficit_type.append(d)
                subset_size.append(s)

    df = pd.DataFrame({
        'epoch': epoch,
        'accuracy': accuracy,
        'deficit_type': deficit_type,
        'subset_size': subset_size
    })

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df,
        x="epoch",
        y="accuracy",
        hue="subset_size",
        style="deficit_type",
        palette="viridis",
        linewidth=2
    )
    plt.title("Deficit Removal During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(title="Subset Size / Deficit Type")
    plt.tight_layout()
    plt.show()
