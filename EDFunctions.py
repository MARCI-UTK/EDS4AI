import torch
import torchvision
import torchvision.transforms.v2 as transforms

import numpy as np
import pandas as pd
import seaborn as sns

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







