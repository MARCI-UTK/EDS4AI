import torch
import torchvision
import numpy as np

from similarity_cosinesim import gen_clip_embeddings, get_class_embeddings, gen_quantiles


if __name__ == "__main__" :
    trainset = torchvision.datasets.CIFAR10("../data", train=True, download=False)
    data = trainset.data
    targets = trainset.targets

    embeds = np.full((50000, 512), -1)

    class_data, ti = get_class_embeddings(embeds, targets, 8)

    # If you want to generate the embeddings run this code:

    #CLIP_embeddings = gen_clip_embeddings(data)
    #np.save("CIFAR10_CLIP_Embeddings.npy", CLIP_embeddings)

    CLIP_embeddings = np.load("CIFAR10_CLIP_Embeddings.npy")

    quantiles = gen_quantiles(CLIP_embeddings,targets,np.arange(0,10))

    np.save("CIFAR10_cosinesim_quantiles.npy", quantiles)