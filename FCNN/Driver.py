
from Trial import Trial

DEFICIT_LIST = ['similarity', 'disimilarity']
#DURATION_LIST = [5, 10, 15, 20, 25, 30, 100, 200]
DURATION_LIST = [5, 10]
#SUBSET_SIZE_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SUBSET_SIZE_LIST = [0.1, 0.2]

POST_EPOCH = 200


if __name__ == '__main__':
    for deficit in DEFICIT_LIST:

        if (deficit == 'similarity') or (deficit == 'disimilarity'):
            all_params = [(i,j) for i in DURATION_LIST for j in SUBSET_SIZE_LIST]

            for params in all_params:

                deficit_duration = params[0]
                subset_size = params[1]
                subset_size_string = str(subset_size).replace('.', '-')

                save_dir = f'{deficit}/epochs_{deficit_duration}_size_{subset_size_string}'
                print(f'save_dir is {save_dir}')

