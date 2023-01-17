import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.lines import Line2D
import matplotlib
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from moseq2_nlp.util import get_unique_list_elements
import umap

all_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'gray', 'pink', 'cyan', 'magenta', 'black']

def dim_red(X, method, **kwargs):
    if method == 'pca':
        pca = PCA(n_components=2)
        z = pca.fit_transform(X)
    elif method == 'tsne':
        z = TSNE(n_components=2, perplexity = kwargs['perplexity']).fit_transform(X)
    elif method == 'umap':
        reducer = umap.UMAP()
        z = reducer.fit_transform(X)
    else:
        raise ValueError('This dimensionality reduction method is not recognized.')

    return z

def visualizer(X, labels, method, save_path, **kwargs):

    unique_labels = get_unique_list_elements(labels)
    colors = all_colors[:len(unique_labels)]

    z = dim_red(X, method, **kwargs) 

    z_colors = [all_colors[unique_labels.index(label)] for label in labels]

    fig, ax = plt.subplots()

    ax.scatter(z[:,0], z[:,1], c=z_colors, alpha=.5)

    legend_elements = [Line2D([0], [0], marker='o', color=color, label=label,
                          markerfacecolor=color, markersize=15, linestyle=None, linewidth=0) for (color, label) in zip(colors, unique_labels)]

    ax.legend(handles=legend_elements)
    
    plt.savefig(save_path)
    plt.close()
