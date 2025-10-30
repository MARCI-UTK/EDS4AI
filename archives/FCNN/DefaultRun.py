import numpy as np
from Trial import Trial

DEFICIT_LIST = ['similarity']
DURATION_LIST = [0]
#DURATION_LIST = [5]
SUBSET_SIZE_LIST = [0.9]
#SUBSET_SIZE_LIST = [0.1, 0.5]

POST_EPOCH = 200
#POST_EPOCH = 2


from pathlib import Path

if __name__ == '__main__':
    for deficit in DEFICIT_LIST:

        if (deficit == 'similarity') or (deficit == 'disimilarity'):
            all_params = [(i,j) for i in DURATION_LIST for j in SUBSET_SIZE_LIST]

            for params in all_params:

                deficit_duration = params[0]
                subset_size = params[1]
                subset_size_string = str(subset_size).replace('.', '-')

                save_dir = f'{deficit}/epochs_{deficit_duration}_size_{subset_size_string}/'
                
                tr_loss, tr_acc, te_loss, te_acc = Trial(deficit=deficit, subset_size=subset_size,
                                                         deficit_duration=deficit_duration, post_duration=POST_EPOCH, save_dir=None)

                #make the directory
                Path(save_dir).mkdir(parents=True, exist_ok=True)

                np.save(save_dir + 'train_losses.npy', tr_loss)
                np.save(save_dir + 'train_accuracies.npy', tr_acc)
                np.save(save_dir + 'test_losses.npy', te_loss)
                np.save(save_dir + 'test_accuracies.npy', te_acc)
                #print(f'save_dir is {save_dir}')

