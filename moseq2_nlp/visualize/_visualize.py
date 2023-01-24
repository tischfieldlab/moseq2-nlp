import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.animation as animation
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from moseq2_nlp.util import get_unique_list_elements
import umap

all_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'gray', 'pink', 'cyan', 'magenta', 'black']

def dim_red(X, method, **kwargs):
    """Reduces the dimensionality of samples, X, by one of three methods. 

        Args: 
            X: a sample x feature numpy array
            method: string indicating which dimensionality reduction method to use: `pca`, `tsne` or `umap`

        Kwargs: 
            perplexity: float controlling the perplexity for the tsne method

        Returns: 
            z: sample x reduced_feature numpy representing the reduced data"""

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

def plot_latent(X, labels, method, save_path, **kwargs):
    """Plots reduced representation of samples from X. 

        Args: 
            X: a sample x feature numpy array
            labels: iterable containing labels for samples from X
            method: string indicating which dimensionality reduction method to use: `pca`, `tsne` or `umap`
            save_path: string controling save destination

        Kwargs: 
            perplexity: float controlling the perplexity for the tsne method

        See also: 
            dim_red """

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

def update_scatter(i, sentence, scat, ax, num_vocab):
    """Helper function for updating the color of a scatter plot

        Args:
            i: integer index to get colors
            sentence: list of integers for accessing colors
            scat: scatter plot object
            ax: plot axis
            num_vocab: number of points in scatter plot

        Return:
            scat: updated scatter object
            ax: updated axis object

        See also:
            animate_latent_path
    """

    colors = ['gray'] * num_vocab
    colors[sentence[i]] = 'red'
    scat.set_facecolors(colors)
    ax.set_title(f'Syllable {sentence[i]}, step {i}', fontsize=36)
    return (scat, ax)

def animate_latent_path(X, sentence, method, save_path, **kwargs):
    """Animated the path of a sentence through a latent space.

        Args: 
            X: a sample x feature numpy array
            sentence: list of integers for accessing colors
            method: string indicating which dimensionality reduction method to use: `pca`, `tsne` or `umap`
            save_path: string controling save destination

        Kwargs:
            perplexity: float controlling the perplexity for the tsne method
        
        See also:
            update_scatter
    """
    num_frames = len(sentence)
    num_vocab  = X.shape[0]
    z = dim_red(X, method, **kwargs)

    fig, ax = plt.subplots(figsize=(16,16))
    fig.patch.set_facecolor('white')

    colors = ['gray'] * num_vocab
    colors[sentence[0]] = 'red'

    scat = plt.scatter(z[:,0], z[:,1], c=colors, s=128, alpha=.5)

    ax.set_title(f'Syllable {sentence[0]}, step {0}', fontsize=36)
    ax.set_xlabel(f'{method} 1', fontsize=24)
    ax.set_ylabel(f'{method} 2', fontsize=24)

    ani = animation.FuncAnimation(fig, update_scatter, frames=range(num_frames),
                                  fargs=(sentence, scat, ax, num_vocab))

    writergif = animation.PillowWriter(fps=1)
    ani.save(save_path, writer=writergif)
