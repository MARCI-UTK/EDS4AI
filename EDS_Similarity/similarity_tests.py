
import numpy as np
import os

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def load_images(dir):
    images = []
    for filename in os.listdir(dir):
        img = Image.open(dir + "/" + filename)#.load()
        images.append(img)
        #print(filename)

    return images

def conv_imglist_to_ndarray(images, size=(224,224)):
    processed = [img.convert('RGB').resize(size) for img in images]

    ndarrays = [np.array(img) for img in processed]

    array = np.stack(ndarrays, axis=0)
    return array
    

def compute_matrix(vector, func):
    size = len(vector)
    matrix = np.empty((size,size))
    for i, elem1 in enumerate(vector):
        for j, elem2 in enumerate(vector):
            matrix[i][j] = func(elem1, elem2)

    return matrix

def compute_matrix_dreamsim(vector ):
    model, preprocess = dreamsim(pretrained=True, device="cpu")
    size = len(vector)
    matrix = np.empty((size,size))
    for i, elem1 in enumerate(vector):
        for j, elem2 in enumerate(vector):
            img1 = preprocess(elem1)
            img2 = preprocess(elem2)
            matrix[i][j] = model(img1, img2)

    return matrix

from similarity_cosinesim import gen_clip_embeddings
def compute_matrix_clip(images):
    img_ndarray = conv_imglist_to_ndarray(images)
    embeds = gen_clip_embeddings(img_ndarray)
    size = len(images)

    similarity_matrix = cosine_similarity(embeds)
    distance_matrix = 1 - similarity_matrix

    #matrix = np.empty((size,size))
    #print(f"matrix shape: {matrix.shape}")
    #print(f"embeds shape: {embeds.shape}")

    #for i, elem1 in enumerate(embeds):
        #for j, elem2 in enumerate(embeds):
            #print(f"embeds 1 shape: {elem1.shape}")
            #print(f"embeds 2 shape: {elem2.shape}")
            #matrix[i][j] = 1 - cosine_similarity(elem1, elem2)


    return distance_matrix 

from dreamsim import dreamsim
def dreamsim_distance(img1, img2):
    model, preprocess = dreamsim(pretrained=True, device="cpu")
    img1_processed = preprocess(img1)
    img2_processed = preprocess(img2)

    distance = model(img1_processed, img2_processed)
    return distance



#def load_images(image_paths, size=(224, 224)):
    #"""
    #Load and resize images, return as flattened arrays + thumbnails.
    #"""
    #image_vectors = []
    #thumbnails = []
    #for path in image_paths:
        #img = Image.open(path).convert('RGB').resize(size)
        #img_array = np.array(img).astype(np.float32).flatten()
        #image_vectors.append(img_array)
        #thumbnails.append(img)
    #return np.stack(image_vectors), thumbnails

def compute_similarity_matrix(image_vectors):
    return cosine_similarity(image_vectors)

def plot_similarity_matrix_with_values(sim_matrix, thumbnails, sizes=(0.2, 20), title="image similarity"):
    thumbnails = [img.convert('RGB').resize((224,224)) for img in thumbnails]

    scale = sizes[0]
    fsize = sizes[1]

    n = sim_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Remove default tick labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Offsets for thumbnails
    HORIZONTAL_OFFSET = -1.0   # Above first row
    VERTICAL_OFFSET = -1.0     # Left of first column

    # Expand limits so images aren't clipped
    ax.set_xlim(VERTICAL_OFFSET - 0.5, n - 0.5)
    ax.set_ylim(HORIZONTAL_OFFSET - 0.5, n - 0.5)

    # Add horizontal thumbnails above matrix
    for i, thumbnail in enumerate(thumbnails):
        imagebox = OffsetImage(thumbnail, zoom=scale)
        ab = AnnotationBbox(
            imagebox,
            (i, HORIZONTAL_OFFSET),
            frameon=False,
            xycoords='data',
            box_alignment=(0.5, 0.5)
        )
        ax.add_artist(ab)

    # Add vertical thumbnails left of matrix
    for i, thumbnail in enumerate(thumbnails):
        imagebox = OffsetImage(thumbnail, zoom=scale)
        ab = AnnotationBbox(
            imagebox,
            (VERTICAL_OFFSET, i),
            frameon=False,
            xycoords='data',
            box_alignment=(0.5, 0.5)
        )
        ax.add_artist(ab)

    # Normalize similarity to [0, 1] for colormap
    cmap = matplotlib.cm.get_cmap('viridis_r')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    # Add numeric similarity values with larger font
    for i in range(n):
        for j in range(n):
            val = sim_matrix[i, j]
            color = cmap(norm(val))
            ax.text(
                j, i, f"{val:.2f}",
                ha='center',
                va='center',
                color='black',              # always black text
                fontsize=fsize,
                fontweight='bold',
                bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.2')
            )
            #ax.text(
                #j, i,
                #f"{val:.2f}",
                #ha='center',
                #va='center',
                #color=color,
                #fontsize=fsize,     # 🔥 Bigger text here!
                #fontweight='bold'
            #)

    ax.set_title(title, fontsize=18)

    plt.show()