
import torchvision

from EDFunctions import *

trainset = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                        download=True, transform=transforms.ToTensor())

testset = torchvision.datasets.CIFAR10(root='./data/', train=False,
                                        download=True, transform=transforms.ToTensor())

data = trainset.data
labels = np.array(range(0, 10))
targets = trainset.targets

embeds = gen_2d_embeddings(data)
#np.save("CIFAR10_train_2d_embeddings.npy", embeds)

quantiles = get_all_quantiles(embeds, targets, labels, 0, 0, True)

np.save("CIFAR10_train_quantiles.npy", quantiles)


#def gen_2d_embeddings(images):
    #model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    #image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    #image_embeddings = np.empty((0,512))

    ## number of images is too large to run clip on all of them so we have to do it in 1000 image batches
    #for i in range(0, len(images), 1000):
        #images_slice = images[ i: i+1000 ][:][:][:]

        #clip_inputs = image_processor(images=images_slice, return_tensors="pt")
        #clip_outputs = model.get_image_features(**clip_inputs).detach()

        #image_embeddings = np.append(image_embeddings, clip_outputs, axis=0)

        #print(i)

    #pca = PCA(n_components=50)
    #pca_results = pca.fit_transform(image_embeddings)

    #tsne = TSNE(n_components=2)
    #tsne_results = tsne.fit_transform(pca_results)

    #return tsne_results


#def get_all_quantiles(embeddings, targets, labels, percentage, where, transform=True):
    #quantiles = np.full((len(targets)), -1, dtype=float)
    #for l in labels:
        #class_data, true_indexes = get_class_embeddings(embeddings, targets, l)
        #num_class_points = class_data.shape[0]

        #mean, _, standardized_data = fit_distribution_3(class_data, transform)

        #distance_list = []
        #for i in range(num_class_points):
            #distance = calc_distance(standardized_data[i, :],  mean)
            #distance_list.append( (distance, true_indexes[i]) )

        #sorted_distances = sorted(distance_list, key=lambda x: x[0])

        #prev_index = 0
        #for i in np.arange(0.001, 1.001, 0.001):
            #list_index = round(i * num_class_points)
            #quantile_points = sorted_distances[prev_index:list_index]

            #for point in quantile_points:
                #true_index = point[1]
                #quantiles[true_index] = i

            #prev_index = list_index

    #return quantiles