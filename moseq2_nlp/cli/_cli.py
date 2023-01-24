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
from moseq2_nlp.data import make_phrases_dataset
from moseq2_nlp.gridsearch import (find_gridsearch_results,
                                   generate_grid_search_worker_params,
                                   get_gridsearch_default_scans,
                                   wrap_command_with_local,
                                   wrap_command_with_slurm, write_jobs)
from moseq2_nlp.util import (IntChoice, command_with_config, ensure_dir,
                              get_command_defaults, write_yaml, get_unique_list_elements)
from moseq2_viz.util import parse_index
from moseq2_viz.model.util import parse_model_results
from moseq2_nlp.visualize import plot_latent, animate_latent_path
import pdb

# Here we will monkey-patch click Option __init__
# in order to force showing default values for all options
orig_init = click.core.Option.__init__

def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True

click.core.Option.__init__ = new_init # type: ignore

@click.group()
@click.version_option()
def cli():
    pass

@cli.command(name='train', cls=command_with_config('config_file'), help='train a classifier')
@click.option('--name', type=str)
@click.option('--save-dir', type=str, default=os.getcwd())
@click.option('--data-path', type=click.Path(exists=True))
@click.option('--representation', type=click.Choice(['embeddings', 'usages', 'transitions']), default='embeddings')
@click.option('--classifier', type=click.Choice(['logistic_regressor', 'svm']), default='logistic_regressor')
@click.option('--kernel', type=click.Choice(['linear', 'poly', 'rbf', 'sigmoid']), default='rbf')
@click.option('--emissions', is_flag=True)
@click.option('--num-syllables', type=int, default=70)
@click.option('--num-transitions', type=int, default=300)
@click.option('--min-count', type=int, default=1)
@click.option('--negative', type=int, default=5)
@click.option('--dm', default=2, type=IntChoice([0, 1, 2]))
@click.option('--embedding-dim', type=int, default=70)
@click.option('--embedding-window', type=int, default=4)
@click.option('--embedding-epochs', type=int, default=250)
@click.option('--bad-syllables', type=int, multiple=True, default=[-5])
@click.option('--test-size', type=float, default=.33)
@click.option('--k', type=int, default=1)
@click.option('--penalty', default='l2', type=click.Choice(['l1', 'l2', 'elasticnet']))
@click.option('--num-c', type=int, default=11)
@click.option('--multi_class', default='ovr', type=click.Choice(['ovr', 'auto', 'multinomial']))
@click.option('--seed', type=int, default=0)
@click.option('--split-seed', type=int, default=0)
@click.option('--verbose', type=int, default=0)
@click.option('--config-file', type=click.Path())
def train(name, save_dir, data_path, representation, classifier, kernel, emissions, num_syllables, num_transitions, min_count, negative, dm, embedding_dim, embedding_window,
          embedding_epochs, bad_syllables, test_size, k, penalty, num_c, multi_class, seed, split_seed, verbose, config_file):
    trainer.train(name, save_dir, data_path, representation, classifier, emissions,  num_syllables, num_transitions, min_count, negative, dm, embedding_dim, embedding_window,
          embedding_epochs, bad_syllables, test_size, k, penalty, num_c, multi_class, kernel, seed, split_seed, verbose)

@cli.command(name="generate-train-config", help="Generates a configuration file that holds editable options for training parameters.")
@click.option('--output-file', '-o', type=click.Path(), default='train-config.yaml')
def generate_train_config(output_file):
    output_file = os.path.abspath(output_file)
    write_yaml(output_file, get_command_defaults(train))
    print(f'Successfully generated train config file at "{output_file}".')

@cli.command(name='make-phrases', help='finds and saves compound modules')
@click.argument('data-path', type=click.Path(exists=True))
@click.option('--save-path', type=click.Path(), default='./all_phrases.pkl')
@click.option('--threshes', type=float, multiple=True, default=[.001,.001])
@click.option('--n', type=int, default=2)
@click.option('--min-count', type=int, default=2)

def make_phrases(data_path, save_path, wordcloud_path, threshes, n, min_count, visualize, max_plot):

    with open(os.path.join(data_path,'sentences.pkl'),'rb') as fn:
        sentences = pickle.load(fn)

    with open(os.path.join(data_path,'labels.pkl'),'rb') as fn:
        labels = pickle.load(fn)

    make_phrases_dataset(sentences, labels, save_path, threshes, n, min_count)
    
@cli.command(name='grid-search', help='grid search hyperparameters')
@click.argument("scan_file", type=click.Path(exists=True))
@click.option('--save-dir', type=click.Path(), default=os.path.join(os.getcwd(), 'worker-configs'), help="Directory to save worker configurations")
@click.option('--cluster-type', default='local', type=click.Choice(['local', 'slurm']))
@optgroup.group('SLURM Scheduler Options', help="The following parameters affect how SLURM jobs are requested, ignored unless --cluster-type=slrum")
@optgroup.option('--slurm-partition', type=str, default='main', help="Partition on which to run jobs. Only for SLURM")
@optgroup.option('--slurm-ncpus', type=int, default=1, help="Number of CPUs per job. Only for SLURM")
@optgroup.option('--slurm-memory', type=str, default="2GB", help="Amount of memory per job. Only for SLURM")
@optgroup.option('--slurm-wall-time', type=str, default='6:00:00', help="Max wall time per job. Only for SLURM")
@optgroup.option('--slurm-preamble', type=str, default='', help="Extra commands to run prior to executing job. Useful for activating an environment, if needed")
@optgroup.option('--slurm-extra', type=str, default='', help="Extra parameters to pass to surm.")
def grid_search(scan_file, save_dir, cluster_type, slurm_partition, slurm_ncpus, slurm_memory, slurm_wall_time, slurm_preamble, slurm_extra):

    worker_dicts = generate_grid_search_worker_params(scan_file)

    if cluster_type == 'local':
        cluster_wrap = wrap_command_with_local
    elif cluster_type == 'slurm':
        cluster_wrap = partial(wrap_command_with_slurm,
                               preamble=slurm_preamble,
                               partition=slurm_partition,
                               ncpus=slurm_ncpus,
                               memory=slurm_memory,
                               wall_time=slurm_wall_time,
                               extra_params=slurm_extra)
    else:
        raise ValueError(f'Unsupported cluster-type {cluster_type}')

    save_dir = ensure_dir(save_dir)
    write_jobs(worker_dicts, cluster_wrap, save_dir)
    sys.stderr.write(f'{len(worker_dicts)} jobs written to {save_dir}\n')


@cli.command(name="generate-gridsearch-config", help="Generates a configuration file that holds editable options for gridsearching hyperparameters.")
@click.option('--output-file', '-o', type=click.Path(), default='gridsearch-config.yaml')
def generate_gridsearch_config(output_file):

    params = {
        'scans': get_gridsearch_default_scans(),
        'parameters': get_command_defaults(train)
    }

    output_file = os.path.abspath(output_file)
    write_yaml(output_file, params)
    print(f'Successfully generated gridsearch config file at "{output_file}".')


@cli.command(name="aggregate-gridsearch-results", help="Aggregate Gridsearch results.")
@click.argument("results-directory", type=click.Path(exists=True))
@click.option("--best-key", type=str, default='best_accuracy')
def aggregate_gridsearch_results(results_directory, best_key):
    
    results = find_gridsearch_results(results_directory).sort_values(best_key, ascending=False)
    results.to_csv(os.path.join(results_directory, 'gridsearch-aggregate-results.tsv'), sep='\t', index=False)

    print('Best model:')
    print(results.iloc[0])

@cli.command(name="moseq-to-raw", help="convert model and index file to raw sentences and labels")
@click.argument("model-file", type=click.Path(exists=True))
@click.argument("index-file", type=click.Path(exists=True))
@click.option("--data-dir", type=str, default='.')
@click.option('--custom-groupings', type=str, multiple=True, default=[])
def moseq_to_raw(model_file, index_file, data_dir):

    ensure_dir(data_dir)

    sentences, labels = get_raw(model_file, index_file, custom_groupings)
    unique_labels = get_unique_list_elements(labels)

    for dat, fn in zip([sentences, labels],['sentences', 'labels']):
        fn = os.path.join(data_dir, f'{fn}.pkl')
        with open(fn, 'wb') as file:
            pickle.dump(dat,file)

@cli.command(name="plot-latent", help="plot latent space of classified data (e.g. animals)")
@click.argument("features_path", type=click.Path(exists=True))
@click.argument("labels_path", type=click.Path(exists=True))
@click.option("--method", type=click.Choice(['pca', 'tsne', 'umap']), default='pca')
@click.option("--save_path", type=click.Path(exists=True), default='./z.png')
@click.option("--perplexity", type=float, default=3.0)
def plot_latent_cmd(features_path, labels_path, method, save_path, perplexity):

    with open(features_path,'rb') as fn:
        X = pickle.load(fn)

    with open(labels_path,'rb') as fn:
        labels = pickle.load(fn)

    plot_latent(X, labels, method, save_path, perplexity=perplexity)

@cli.command(name="animate-latent", help="animate path of unclassified data (e.g. syllables)")
@click.argument("features_path", type=click.Path(exists=True))
@click.argument("model-file", type=click.Path(exists=True))
@click.argument("index-file", type=click.Path(exists=True))
@click.argument("animal-index", type=int)
@click.option("--method", type=click.Choice(['pca', 'tsne', 'umap']), default='pca')
@click.option("--save_path", type=click.Path(exists=True), default='./z_anim.gif')
@click.option("--perplexity", type=float, default=3.0)
def animate_latent_cmd(features_path, model_file, index_file, method, save_path, perplexity):

    with open(features_path,'rb') as fn:
        X = pickle.load(fn)

    sentences, _ = get_raw(model_file, index_file, custom_groupings)

    sentence = sentence[animal_index]

    animate_latent_path(X, sentence, method, save_path, perplexity=perplexity)

if __name__ == '__main__':
    cli()
