import os
import sys
from functools import partial
import pickle
from typing import List

import click
from click_option_group import optgroup
from numpy.random import choice, randint
from tqdm import tqdm

import moseq2_nlp.train as trainer
from moseq2_nlp.data import save_phrase_datasets, save_brown_datasets, get_emissions
from moseq2_nlp.explain import Explainer

from moseq2_nlp.gridsearch import (
    find_gridsearch_results,
    generate_grid_search_worker_params,
    get_gridsearch_default_scans,
    wrap_command_with_local,
    wrap_command_with_slurm,
    write_jobs,
)
from moseq2_nlp.util import IntChoice, command_with_config, ensure_dir, get_command_defaults, write_yaml, get_unique_list_elements
from moseq2_viz.util import parse_index
from moseq2_viz.model.util import parse_model_results
from moseq2_nlp.visualize import plot_latent, animate_latent_path

# Here we will monkey-patch click Option __init__
# in order to force showing default values for all options
orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    """Sets click init."""
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init  # type: ignore


@click.group()
@click.version_option()
def cli():
    """Entry point."""
    pass


@cli.command(name="train", cls=command_with_config("config_file"), help="train a classifier")
@click.option("--name", type=str)
@click.option("--save-dir", type=str, default=os.getcwd())
@click.option("--data-path", type=click.Path(exists=True))
@click.option("--representation", type=click.Choice(["embeddings", "usages", "transitions"]), default="embeddings")
@click.option("--classifier", type=click.Choice(["logistic_regressor", "svm"]), default="logistic_regressor")
@click.option("--kernel", type=click.Choice(["linear", "poly", "rbf", "sigmoid"]), default="rbf")
@click.option("--emissions", is_flag=True)
@click.option("--num-syllables", type=int, default=70)
@click.option("--num-transitions", type=int, default=300)
@click.option("--min-count", type=int, default=1)
@click.option("--negative", type=int, default=5)
@click.option("--dm", default=2, type=IntChoice([0, 1, 2]))
@click.option("--embedding-dim", type=int, default=70)
@click.option("--embedding-window", type=int, default=4)
@click.option("--embedding-epochs", type=int, default=250)
@click.option("--bad-syllables", type=int, multiple=True, default=[-5])
@click.option("--test-size", type=float, default=0.33)
@click.option("--k", type=int, default=1)
@click.option("--penalty", default="l2", type=click.Choice(["l1", "l2", "elasticnet"]))
@click.option("--num-c", type=int, default=11)
@click.option("--multi_class", default="ovr", type=click.Choice(["ovr", "auto", "multinomial"]))
@click.option("--max-iter", type=int, default=2000)
@click.option("--seed", type=int, default=0)
@click.option("--split-seed", type=int, default=0)
@click.option("--verbose", type=int, default=0)
@click.option("--config-file", type=click.Path())
def train(
    name,
    save_dir,
    data_path,
    representation,
    classifier,
    kernel,
    emissions,
    num_syllables,
    num_transitions,
    min_count,
    negative,
    dm,
    embedding_dim,
    embedding_window,
    embedding_epochs,
    bad_syllables,
    test_size,
    k,
    penalty,
    num_c,
    multi_class,
    max_iter,
    seed,
    split_seed,
    verbose,
    config_file,
):
    """Train a classifier from scratch.

    Arguments:
        name: str, name of the experiment
        save_dir: str, where model will be saved
        data_path: str, where sentences and labels are stored
        representation: literal ('usages', 'transitions', 'embeddings') indicating feature type
        classifier: literal ('logistic_regressor', 'svm') indicating classifier type
        kernel: literal ('linear', 'poly', 'rbf', 'sigmoid') indicating kernel for svm
        emissions: bool, whether or not to use emissions or frames
        num_syllables: int, max syllables to include in analysis
        num_transitions: int, max number of transitions to include in transition rep
        min_count: int, minimum # of times syllables has to appear overall to be included in analysis
        negative: int, exponent used for negative sampling in doc2vec
        dm: literal (0,1,2) indicating which between cbow, dm or their average to use for doc2vec
        embedding_dim: int, dimension of d2v embedding space
        embedding_window: int, window size for d2v context
        embedding_epochs: int, number of training steps for d2v
        bad_syllables: list of ints, syllables to exclude
        test_size: float, prop for test set
        k: int, k-fold cv
        penalty: literal ('l2', 'l1', 'elasticnet') for regressor penalty temr
        num_c: int, number of regularizer weights chosen logarithmically between 1e-5 and 1e5 for classifier
        multi_class: literal ('ovr', 'auto', 'multinomial'), multiclass scheme for logistic regressor
        max_iter: integer, maximum number of steps for classifier training
        seed: int, seed for features
        split_seed: int, seed for train-test split
        verbose: int, 0 for no messages
        config_file: str, path for config variables.

    """
    trainer.train(
        name,
        save_dir,
        data_path,
        representation,
        classifier,
        emissions,
        num_syllables,
        num_transitions,
        min_count,
        negative,
        dm,
        embedding_dim,
        embedding_window,
        embedding_epochs,
        bad_syllables,
        test_size,
        k,
        penalty,
        num_c,
        multi_class,
        kernel,
        max_iter,
        seed,
        split_seed,
        verbose,
    )


@cli.command(name="generate-train-config", help="Generates a configuration file that holds editable options for training parameters.")
@click.option("--output-file", "-o", type=click.Path(), default="train-config.yaml")
def generate_train_config(output_file):
    """Generate a train config file.

    Args:
        output_file: str, where to save the config yaml.

    """
    output_file = os.path.abspath(output_file)
    write_yaml(output_file, get_command_defaults(train))
    print(f'Successfully generated train config file at "{output_file}".')


@cli.command(name="make-phrases", help="finds and saves compound modules")
@click.argument("data-path", type=click.Path(exists=True))
@click.option("--save-dir", type=click.Path(), default="./phrases")
@click.option("--emissions", is_flag=True)
@click.option("--thresholds", type=float, multiple=True, default=[0.1])
@click.option("--iterations", type=int, default=1)
@click.option("--min-count", type=int, default=1)
@click.option("--scoring", type=str, default="default")
def make_phrases(data_path, save_dir, emissions, thresholds, iterations, min_count, scoring):
    """Detect phrases and save them in a dict.

    Args:
        data_path: str, where sentences and labels are saved
        save_dir: str, where to save the phrases dict
        emissions: bool, if true, use emissions
        thresholds: list of floats, indicating threshholds for phrase detection
        iterations: int, number of times to aggregate syllables into higher-order units
        min_count: int, minimum # of times syllables has to appear overall to be included in analysis.
        scoring: str, which scoring method to use, either 'default' or 'npmi'
    """
    ensure_dir(save_dir)

    with open(os.path.join(data_path, "sentences.pkl"), "rb") as fn:
        sentences = pickle.load(fn)
        if emissions:
            sentences = get_emissions(sentences)

    with open(os.path.join(data_path, "labels.pkl"), "rb") as fn:
        labels = pickle.load(fn)

    save_phrase_datasets(sentences, labels, thresholds, save_dir, iterations=iterations, min_count=min_count, scoring=scoring)


@cli.command(name="make-synonyms", help="finds and saves module clusters with Brown clustering")
@click.argument("data-path", type=click.Path(exists=True))
@click.argument("save-dir", type=click.Path(), default="./brown_synonyms")
@click.option("--emissions", is_flag=True)
@click.option("--alpha", type=float, default=0.5)
@click.option("--min-count", type=int, default=0)
def make_synonyms(data_path, save_dir, emissions, alpha, min_count):
    """Detect Brown synonyms and save them according to different resolutions in save_dir.

    Args:
        data_path: str, where sentences and labels are saved
        save_dir: str, where to save the synonym dict
        emissions: bool, if true, use emissions
        alpha: float, Laplacian smoothing coefficient for Brown clusterer
        min_count: int, minimum # of times syllables has to appear overall to be included in analysis.

    """
    ensure_dir(save_dir)

    with open(os.path.join(data_path, "sentences.pkl"), "rb") as fn:
        sentences = pickle.load(fn)
        if emissions:
            sentences = get_emissions(sentences)

    with open(os.path.join(data_path, "labels.pkl"), "rb") as fn:
        labels = pickle.load(fn)

    save_brown_datasets(sentences, labels, save_dir, alpha, min_count)


@cli.command(name="grid-search", help="grid search hyperparameters")
@click.argument("scan_file", type=click.Path(exists=True))
@click.option(
    "--save-dir", type=click.Path(), default=os.path.join(os.getcwd(), "worker-configs"), help="Directory to save worker configurations"
)
@click.option("--cluster-type", default="local", type=click.Choice(["local", "slurm"]))
@optgroup.group(
    "SLURM Scheduler Options", help="The following parameters affect how SLURM jobs are requested, ignored unless --cluster-type=slrum"
)
@optgroup.option("--slurm-partition", type=str, default="main", help="Partition on which to run jobs. Only for SLURM")
@optgroup.option("--slurm-ncpus", type=int, default=1, help="Number of CPUs per job. Only for SLURM")
@optgroup.option("--slurm-memory", type=str, default="2GB", help="Amount of memory per job. Only for SLURM")
@optgroup.option("--slurm-wall-time", type=str, default="6:00:00", help="Max wall time per job. Only for SLURM")
@optgroup.option(
    "--slurm-preamble",
    type=str,
    default="",
    help="Extra commands to run prior to executing job. Useful for activating an environment, if needed",
)
@optgroup.option("--slurm-extra", type=str, default="", help="Extra parameters to pass to surm.")
def grid_search(
    scan_file, save_dir, cluster_type, slurm_partition, slurm_ncpus, slurm_memory, slurm_wall_time, slurm_preamble, slurm_extra
):
    """Write jobs for gridsearch.

    Args:
        scan_file: str, config yaml with gridsearch parameters
        save_dir: str, where to save the job strings
        cluster_type: str, either `local` or `slurm`
        slurm_partition: str, Partition on which to run jobs. Only for SLURM
        slurm_ncpus: int, Number of CPUs per job. Only for SLURM
        slurm_memory: str, Amount of memory per job. Only for SLURM
        slurm_wall_time: str, Max wall time per job. Only for SLURM
        slurm_preamble: str, Extra commands to run prior to executing job. Useful for activating an environment, if needed
        slurm_extra: str, Extra parameters to pass to slurm.

    """
    worker_dicts = generate_grid_search_worker_params(scan_file)

    if cluster_type == "local":
        cluster_wrap = wrap_command_with_local
    elif cluster_type == "slurm":
        cluster_wrap = partial(
            wrap_command_with_slurm,
            preamble=slurm_preamble,
            partition=slurm_partition,
            ncpus=slurm_ncpus,
            memory=slurm_memory,
            wall_time=slurm_wall_time,
            extra_params=slurm_extra,
        )
    else:
        raise ValueError(f"Unsupported cluster-type {cluster_type}")

    save_dir = ensure_dir(save_dir)
    write_jobs(worker_dicts, cluster_wrap, save_dir)
    sys.stderr.write(f"{len(worker_dicts)} jobs written to {save_dir}\n")


@cli.command(
    name="generate-gridsearch-config", help="Generates a configuration file that holds editable options for gridsearching hyperparameters."
)
@click.option("--output-file", "-o", type=click.Path(), default="gridsearch-config.yaml")
def generate_gridsearch_config(output_file):
    """Generate gridsearch config file.

    Args:
        output_file:str, where to save the config.

    """
    params = {"scans": get_gridsearch_default_scans(), "parameters": get_command_defaults(train)}

    output_file = os.path.abspath(output_file)
    write_yaml(output_file, params)
    print(f'Successfully generated gridsearch config file at "{output_file}".')


@cli.command(name="aggregate-gridsearch-results", help="Aggregate Gridsearch results.")
@click.argument("results-directory", type=click.Path(exists=True))
@click.option("--best-key", type=str, default="best_accuracy")
def aggregate_gridsearch_results(results_directory, best_key):
    """Aggregate gridsearch results.

    Args:
        results_directory: str, path where results are saved.
        best_key: str, how to sort results.

    """
    results = find_gridsearch_results(results_directory).sort_values(best_key, ascending=False)
    results.to_csv(os.path.join(results_directory, "gridsearch-aggregate-results.tsv"), sep="\t", index=False)

    print("Best model:")
    print(results.iloc[0])


@cli.command(name="moseq-to-raw", help="convert model and index file to raw sentences and labels")
@click.argument("model-file", type=click.Path(exists=True))
@click.argument("index-file", type=click.Path(exists=True))
@click.option("--data-dir", type=str, default=".")
@click.option("--custom-groupings", type=str, multiple=True, default=[])
def moseq_to_raw(model_file, index_file, data_dir):
    """Convert model file and index file to sentences and labels.

    Args:
        model_file: p file from moseq
        index_file: yaml file from moseq
        data_dir: str, where to save sentences, labels
        custom_groupings: list of str, each str is a comma separated sequence of labels to be grouped into one class

    """
    ensure_dir(data_dir)

    sentences, labels = get_raw(model_file, index_file, custom_groupings)
    unique_labels = get_unique_list_elements(labels)

    for dat, fn in zip([sentences, labels], ["sentences", "labels"]):
        fn = os.path.join(data_dir, f"{fn}.pkl")
        with open(fn, "wb") as file:
            pickle.dump(dat, file)


@cli.command(name="plot-latent", help="plot latent space of classified data (e.g. animals)")
@click.argument("features-path", type=click.Path(exists=True))
@click.argument("labels-path", type=click.Path(exists=True))
@click.option("--method", type=click.Choice(["pca", "tsne", "umap"]), default="pca")
@click.option("--save-path", default="./z.png")
@click.option("--perplexity", type=float, default=3.0)
def plot_latent_cmd(features_path, labels_path, method, save_path, perplexity):
    """Plot pca/tsne/umap of features.

    Args:
        features_path: str, where features are saved
        labels_path: str, where labels are saved
        method: str, which method (pca, tsne, map) used for dim reduction
        save_path: str, where to save plot
        perplexity: float, perplexity argument for tsne method

    """
    with open(features_path, "rb") as fn:
        X = pickle.load(fn)

    with open(labels_path, "rb") as fn:
        labels = pickle.load(fn)

    plot_latent(X, labels, method, save_path, perplexity=perplexity)


@cli.command(name="animate-latent", help="animate path of unclassified data (e.g. syllables)")
@click.argument("features-path", type=click.Path(exists=True))
@click.argument("model-file", type=click.Path(exists=True))
@click.argument("index-file", type=click.Path(exists=True))
@click.argument("animal-index", type=int)
@click.option("--method", type=click.Choice(["pca", "tsne", "umap"]), default="pca")
@click.option("--save_path", type=click.Path(exists=True), default="./z_anim.gif")
@click.option("--perplexity", type=float, default=3.0)
def animate_latent_cmd(features_path, model_file, index_file, method, save_path, perplexity):
    """Animate sequence of syllables in latent space (pca, tsne, umap).

    Args:
        features_path: str, where features are saved
        model_file: p file from moseq
        index_file: yaml file from moseq
        animal_index: int, index of animal whose path to visualize
        method: str, which method (pca, tsne, map) used for dim reduction
        save_path: str, where to save animation
        perplexity: float, perplexity argument for tsne method.

    """
    with open(features_path, "rb") as fn:
        X = pickle.load(fn)

    sentences, _ = get_raw(model_file, index_file, custom_groupings)

    sentence = sentence[animal_index]

    animate_latent_path(X, sentence, method, save_path, perplexity=perplexity)


@cli.command(name="explain-classifier", help="explains the results of a classifier")
@click.argument("data-path", type=click.Path(exists=True))
@click.argument("clf-path", type=click.Path(exists=True))
@click.argument("feature-name", type=click.Choice(["usages", "transitions", "embeddings"]))
@click.option("--save-dir", type=str, default=".")
@click.option("--num-samples", type=int, default=1000)
@click.option("--max-syllable", type=int, default=70)
@click.option("--bad-syllables", type=int, multiple=True, default=[-5])
@click.option("--num-transitions", type=int, default=70)
@click.option("--emissions", is_flag=True)
@click.option("--min-count", type=int, default=1)
@click.option("--negative", type=int, default=5)
@click.option("--dm", default=2, type=IntChoice([0, 1, 2]))
@click.option("--embedding-dim", type=int, default=70)
@click.option("--embedding-window", type=int, default=4)
@click.option("--embedding-epochs", type=int, default=250)
@click.option("--seed", type=int, default=0)
@click.option("--model_path", type=str, default="./dv.model")
def explain_classifier(
    data_path,
    clf_path,
    feature_name,
    save_dir,
    num_samples,
    max_syllable,
    bad_syllables,
    num_transitions,
    emissions,
    min_count,
    negative,
    dm,
    embedding_dim,
    embedding_window,
    embedding_epochs,
    seed,
    model_path,
):
    """Load a saved classifier and explain classifications using LIME.

    Args:
        data_path: str, path to dir where sentences and labels are saved.
        clf_path: str, path to saved classifier
        feature_name: str, type of feature map, one of `usages`, `transitions` or `embeddings`.
        num_samples: int, number of samples for LIME.
        save_dir: str, dir to save explanation results.
        emissions: bool, whether or not to use emissions or frames
        max_syllable: int, max syllables to include in analysis
        num_transitions: int, max number of transitions to include in transition rep
        min_count: int, minimum # of times syllables has to appear overall to be included in analysis
        negative: int, exponent used for negative sampling in doc2vec
        dm: literal (0,1,2) indicating which between cbow, dm or their average to use for doc2vec
        embedding_dim: int, dimension of d2v embedding space
        embedding_window: int, window size for d2v context
        embedding_epochs: int, number of training steps for d2v
        bad_syllables: list of ints, syllables to exclude
        seed: int, seed for features
        model_path: str, path where the dv model is saved.

    """
    # Load data
    with open(os.path.join(data_path, "sentences.pkl"), "rb") as fn:
        sentences = pickle.load(fn)
    if emissions:
        sentences = get_emissions(sentences)

    with open(os.path.join(data_path, "labels.pkl"), "rb") as fn:
        labels = pickle.load(fn)

    string_sentences = [" ".join(item for item in sentence) for sentence in sentences]

    # Inference feature map args
    if feature_name == "usages":
        kwargs = {"max_syllable": max_syllable, "bad_syllables": bad_syllables}
    elif feature_name == "transitions":
        kwargs = {"max_syllable": max_syllable, "bad_syllables": bad_syllables, "num_transitions": num_transitions}
    else:
        kwargs = {}

    # Embedding kwargs
    embedding_kwargs = {
        "dm": dm,
        "embedding_dim": embedding_dim,
        "embedding_window": embedding_window,
        "embedding_epochs": embedding_epochs,
        "min_count": min_count,
        "negative": negative,
        "seed": seed,
        "model_path": model_path,
    }
    # Count vocab
    n_vocab = len({s for sentence in sentences for s in sentence})

    # Load classifier
    clf = pickle.load(open(clf_path, "rb"))

    # Initialize LIME explainer
    class_names = list(set(labels))
    exp = Explainer(feature_name, clf, class_names, bow=True, custom_feature_map=None, embedding_kwargs=embedding_kwargs, **kwargs)

    # Explain classes and save
    ds_explanation = exp.explain_dataset(string_sentences, labels, num_features=n_vocab, num_samples=num_samples)
    with open(fn, "wb") as file:
        pickle.dump(ds_explanation, os.path.join(save_dir, "explanation.pkl"))


if __name__ == "__main__":
    cli()
