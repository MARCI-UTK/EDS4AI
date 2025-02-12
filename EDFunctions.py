import torch
import torchvision
import torchvision.transforms.v2 as transforms

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import CLIPModel
from transformers import AutoProcessor

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from collections import OrderedDict


#return a numpy array of shape 60000x3x32x32 for images and a list of class labels
def get_ciafar10_data ():
    dataset1 = torchvision.datasets.CIFAR10(root='./data/', train=False,
                                            download=True, transform=transforms.ToTensor())

    dataset2 = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                            download=True, transform=transforms.ToTensor())


    images = np.append(dataset1.data, dataset2.data, axis=0)
    labels = dataset1.targets + dataset2.targets

    images_transposed = np.transpose(images, (0,3,1,2))

    return images_transposed, labels

# For some reason this code crashes when imported and run
def gen_2d_embeddings(images, num_embeddings):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_embeddings = np.empty((0,512))

    # number of images is too large to run clip on all of them so we have to do it in 1000 image batches
    for i in range(60):
        images_slice = images[ (1000*i): (1000*(i+1)) ][:][:][:]

        clip_inputs = image_processor(images=images_slice, return_tensors="pt")
        clip_outputs = model.get_image_features(**clip_inputs).detach()

        image_embeddings = np.append(image_embeddings, clip_outputs, axis=0)

        print(i)

    pca = PCA(n_components=50)
    pca_results = pca.fit_transform(image_embeddings)

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(pca_results)

    return tsne_results


#def gen_2d_embeddings(clip_embeddings):
    #pca = PCA(n_components=50)
    #pca_results = pca.fit_transform(clip_embeddings)

    #tsne = TSNE(n_components=2)
    #tsne_results = tsne.fit_transform(pca_results)

    #return tsne_results


def calc_distance(coordinate_1, coordinate_2):
    return np.linalg.norm(coordinate_1 - coordinate_2)



def calc_distances(coords):
    num_coords = coords.shape[0]

    distance_lists = []
    for i in range(num_coords):
        coordinate = coords[i]
        distance_dict = {j : calc_distance(coordinate, coords[j]) for j in range(num_coords)}
        distance_lists.append(distance_dict)

    return distance_lists



def sort_distances(distances_list):
    from collections import OrderedDict

    m = len(distances_list)
    sorted_distances = []

    for i in range(m):
        unsorted_dict = distances_list[i]
        sorted_dict = OrderedDict(sorted(unsorted_dict.items(), key=lambda item: item[1], reverse=True))
        sorted_distances.append(sorted_dict)

    return sorted_distances




def generate_indexes(coords, sorted_distances, batch_size):
    m = coords.shape[0]
    batch = []

    #distances = calc_distances(coords)
    #sorted_distances = sort_distances(distances)

    indx = 0
    rng = np.random.default_rng()
    indx = rng.integers(0, m)
    batch.append(indx)
    prev_indx = indx

    for i in range(batch_size-1):
        distance_dict = sorted_distances[prev_indx]

        for indx, distance in distance_dict.items():
            if indx in batch:
                pass
            else :
                batch.append(indx)
                prev_indx = indx
                break

    return batch



def generate_indexes_proportional(coords, sorted_distances, batch_size):
    m = coords.shape[0]
    batch = []

    rng = np.random.default_rng()
    indx = rng.integers(0, m)
    batch.append(indx)
    prev_indx = indx

    for i in range(batch_size-1):
        distance_dict = sorted_distances[prev_indx]

        #remove previously used batch indexes from consideration
        for indx in batch:
            if indx in distance_dict.keys():
                distance_dict.pop(indx)

    
        indxs = list(distance_dict.keys())
        distances = list(distance_dict.values())

        probabilities = np.array(distances) / sum(distances)

        indx = np.random.choice(indxs, p=probabilities)

        batch.append(indx)
        prev_indx = indx

    return batch



def plot_classes_2d(image_embeddings, targets):
    column_vals = ['x1', 'x2']

    df2 = pd.DataFrame(data = image_embeddings, columns=column_vals)

    #targets_slice = targets2[0:slice_size]

    label_list =  ['airplane', 'automobile',
                    'bird',
                    'cat',
                    'deer',
                    'dog',
                    'frog',
                    'horse',
                    'ship',
                    'truck']

    image_labels = [label_list[x] for x in targets]

    print(len(label_list))

    df2['label'] = image_labels

    plot_raw = sns.scatterplot(x='x1', y='x2', data=df2, hue='label', ec=None, palette="deep")

    plot_raw.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    fig = plot_raw.get_figure()

    return fig 

def plot_batch_elements(embeddings, batch_indexes):
    column_vals = ['x1', 'x2']

    df = pd.DataFrame(data = embeddings, columns=column_vals)

    m = embeddings.shape[0]
    chosen = ['not chosen' for x in range(m)]
    for indx in batch_indexes:
        chosen[indx] = 'batch element'

    df['label'] = chosen

    hue_order = ['not chosen', 'batch element']
    df_sorted = df.sort_values('label', key=np.vectorize(hue_order.index))

    plot = sns.scatterplot(x='x1', y='x2', data=df_sorted, hue='label', ec=None, palette="deep")
    plot.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    fig = plot.get_figure()
    return fig


def plot_batch_order(embeddings, batch_indexes, batch_size, type='inclusive'):
    column_vals = ['x1', 'x2']

    df = pd.DataFrame(data = embeddings, columns=column_vals)

    m = embeddings.shape[0]
    chosen = ['not chosen' for x in range(m)]
    for indx in batch_indexes:
        chosen[indx] = 'batch element'

    df['label'] = chosen

    df['batch_order'] = 0

    if (type == 'inclusive'):
        for i, batch_index in enumerate(batch_indexes):
            df.at[batch_index, 'batch_order'] = batch_size + i

        hue_order = ['not chosen', 'batch element']
        df_sorted = df.sort_values('label', key=np.vectorize(hue_order.index))

        plot = sns.scatterplot(x='x1', y='x2', data=df_sorted, hue='batch_order', ec=None, legend='brief')
        plot.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

        fig = plot.get_figure()
        return fig

    elif (type == 'exclusive'):
        for i, batch_index in enumerate(batch_indexes):
            df.at[batch_index, 'batch_order'] = i


        hue_order = ['not chosen', 'batch element']
        df_sorted = df.sort_values('label', key=np.vectorize(hue_order.index))

        df_sorted = df_sorted[df_sorted['label'] == 'batch element']

        plot = sns.scatterplot(x='x1', y='x2', data=df_sorted, hue='batch_order', ec=None, legend='brief')
        plot.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

        fig = plot.get_figure()
        return fig
