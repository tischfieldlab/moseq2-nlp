import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt

plt.style.use("ggplot")
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.animation as animation
import matplotlib.cm as cm
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from moseq2_nlp.util import get_unique_list_elements
import umap
import pandas as pd


def dim_red(X, method, **kwargs):
    """Reduces the dimensionality of samples, X, by one of three methods.

    Args:
        X: a sample x feature numpy array
        method: string indicating which dimensionality reduction method to use: `pca`, `tsne` or `umap`

    Kwargs:
        perplexity: float controlling the perplexity for the tsne method

    Returns:
        z: sample x reduced_feature numpy representing the reduced data
    """
    if method == "pca":
        pca = PCA(n_components=2)
        z = pca.fit_transform(X)
    elif method == "tsne":
        z = TSNE(n_components=2, perplexity=kwargs["perplexity"]).fit_transform(X)
    elif method == "umap":
        reducer = umap.UMAP()
        z = reducer.fit_transform(X)
    else:
        raise ValueError("This dimensionality reduction method is not recognized.")

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

    See Also:
        dim_red
    """
    unique_labels = get_unique_list_elements(labels)

    z = dim_red(X, method, **kwargs)

    fig, ax = plt.subplots()
    legend_elements = []
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        z_subtype = np.array([zz for i, zz in enumerate(z) if labels[i] == label])
        ax.scatter(z_subtype[:, 0], z_subtype[:, 1], color=color, alpha=0.5)

        legend_elements.append(
            Line2D([0], [0], marker="o", color=color, label=label, markerfacecolor=color, markersize=15, linestyle=None, linewidth=0)
        )

    ax.legend(handles=legend_elements)

    plt.savefig(save_path)
    plt.close()


def update_scatter(i, sentence, scat, ax, num_vocab):
    """Helper function for updating the color of a scatter plot.

    Args:
        i: integer index to get colors
        sentence: list of integers for accessing colors
        scat: scatter plot object
        ax: plot axis
        num_vocab: number of points in scatter plot

    Return:
        scat: updated scatter object
        ax: updated axis object

    See Also:
        animate_latent_path
    """
    colors = ["gray"] * num_vocab
    colors[sentence[i]] = "red"
    scat.set_facecolors(colors)
    ax.set_title(f"Syllable {sentence[i]}, step {i}", fontsize=36)
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

    See Also:
        update_scatter
    """
    num_frames = len(sentence)
    num_vocab = X.shape[0]
    z = dim_red(X, method, **kwargs)

    fig, ax = plt.subplots(figsize=(16, 16))
    fig.patch.set_facecolor("white")

    colors = ["gray"] * num_vocab
    colors[sentence[0]] = "red"

    scat = plt.scatter(z[:, 0], z[:, 1], c=colors, s=128, alpha=0.5)

    ax.set_title(f"Syllable {sentence[0]}, step {0}", fontsize=36)
    ax.set_xlabel(f"{method} 1", fontsize=24)
    ax.set_ylabel(f"{method} 2", fontsize=24)

    ani = animation.FuncAnimation(fig, update_scatter, frames=range(num_frames), fargs=(sentence, scat, ax, num_vocab))

    writergif = animation.PillowWriter(fps=1)
    ani.save(save_path, writer=writergif)

def visualize_gridsearch(exp_dir):
    sub_exp_dirs = os.listdir(exp_dir)
    sub_exp_dirs = [dr for dr in sub_exp_dirs if 'pdf' not in dr]
    sub_exp_names = [dr.split('/')[0] for dr in sub_exp_dirs]
    num_exps = len(sub_exp_dirs)
    
    fig, ax = plt.subplots()
    
    for d, dr in enumerate(sub_exp_dirs):
    
        csv_fn = os.path.join(exp_dir, dr, 'gridsearch-aggregate-results.csv')
    
        results = pd.read_csv(csv_fn, sep="\t")
    
        names = results.loc[:, 'name'].tolist()
        train_f1 = results.loc[:,'train_f1-score'].to_numpy()
        test_f1  = results.loc[:,'test_f1-score'].to_numpy()
    
        inds = np.argsort(train_f1)
        sorted_train_f1 = train_f1[inds]
        sorted_test_f1  = test_f1[inds]
        sorted_names    = [names[ind] for ind in inds]
    
        disps  = [-.25, 0, .25]
        colors = ['r', 'b', 'g']
        for r, representation in enumerate(['usages', 'transitions', 'embeddings']):
            rep_sorted_train_f1 = [f1 for i, f1 in enumerate(sorted_train_f1) if representation in sorted_names[i]]
            rep_sorted_test_f1 = [f1 for i, f1 in enumerate(sorted_test_f1) if representation in sorted_names[i]]
    
            max_train_f1 = max(rep_sorted_train_f1)
            # Get ties
            max_inds = [i for i, f1 in enumerate(rep_sorted_train_f1) if f1 == max_train_f1]
            all_max_tests = [rep_sorted_test_f1[ind] for ind in max_inds]
    
            mean = np.mean(all_max_tests)
            std  = np.std(all_max_tests)
    
            x = (d - .5) + disps[r]
            ax.bar(x, mean, color=colors[r], width=.25)
            ax.errorbar(x, mean, yerr=std, ecolor='black', capsize=6)
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='green', lw=4)]
    ax.legend(custom_lines, ['Usages', 'Transitions', 'Embeddings'])
    ax.set_ylabel('Best testing f1')
    ax.set_xticks(np.arange(num_exps)-.5, labels=sub_exp_names)
    plt.savefig(os.path.join(exp_dir, 'bars.pdf'))
