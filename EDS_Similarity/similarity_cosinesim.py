########### HEADER ############
# The purpose of this file is to create helper functions for generating embeddings and 
# and the overall embedding generation function too. 
#
# The pipeline being used here is 
# Cifar 10 images -> 512-D Clip Embeddings -> Generate Mean vector for each class 
# -> for each class, calculate the cosine similarity between each vector and the mean
# -> for each vector rank it's similarity relative to other vectors in the class
#
# This will be the notion of similarity present in the quantiles (the file which
# contains a similarity score for each cifar-10 image)

import numpy as np
import sklearn

from transformers import CLIPModel, AutoProcessor
from sklearn.metrics.pairwise import cosine_similarity


# take in images as num_images x H x W x 3
def gen_clip_embeddings(images):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_embeddings = np.empty((0,512))

    # Amount of images is too large to run clip on all of them so we have to do it in 1000 image batches
    for i in range(0, len(images), 1000):
        images_slice = images[ i: i+1000 ][:][:][:]

        clip_inputs = image_processor(images=images_slice, return_tensors="pt")
        clip_outputs = model.get_image_features(**clip_inputs).detach()

        image_embeddings = np.append(image_embeddings, clip_outputs, axis=0)

        #print(i)

    return image_embeddings 




# This function takes in all the information from the dataset and outputs the class
# specific data specified in the label parameter.
#
# embeddings  --  clip embeddings (M x 512)
# targets  --  label values associated with each embedding (M x 1)
# label  --  the value of the label to filter the data by
#
# Returns: 
# class_data  --  the embeddings for each data point in the specified class (num_data_points x 512)
# true_indexes  --  the true index into the original dataset associated with each element of class_data
def get_class_embeddings(embeddings, targets, label):
    targets = np.array(targets)
    targets = np.expand_dims(targets, 1)

    all_indexes = np.array(range(0, embeddings.shape[0]))
    all_indexes = np.expand_dims(all_indexes, 1)

    full_data = np.append(embeddings, all_indexes, axis=1) 
    full_data = np.append(full_data, targets, axis=1) 

    class_data = full_data[ full_data[:,-1] == label ]
    true_indexes = class_data[:,-2].astype(int)
    class_data = class_data[:,:-2]


    return class_data, true_indexes





# clip embeddings  -- are the output from gen_clip_embeddings (e.g 50000x512)
# targets  --  are the associated label value (e.g 50000x512)
#    they should line up with the clip embeddings
# label_names  --  is a list of the possible values for each target, 
#    (e.g. for 10 classes it will be a list of length 10)

def gen_quantiles(clip_embeddings, targets, labels_list):
    quantiles = np.full(len(targets), -1, dtype=float)
    for l in labels_list:
        class_data, true_indexes = get_class_embeddings(clip_embeddings, targets, l)
        num_class_points = class_data.shape[0]

        #mean, _, standardized_data = fit_distribution_3(class_data, transform)
        mean_embedding = np.mean(class_data, axis=0)

        distance_list = []
        for i in range(num_class_points):
            distance = cosine_similarity( [class_data[i, :]] ,  [mean_embedding])
            distance_list.append( (distance, true_indexes[i]) )

        sorted_distances = sorted(distance_list, key=lambda x: x[0])

        prev_index = 0
        for i in np.arange(0.001, 1.001, 0.001):
            list_index = round(i * num_class_points)
            quantile_points = sorted_distances[prev_index:list_index]

            for point in quantile_points:
                true_index = point[1]
                quantiles[true_index] = i

            prev_index = list_index

    return quantiles