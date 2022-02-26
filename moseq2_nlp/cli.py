import os
import sys
from functools import partial

import click
from numpy.random import randint, choice

import moseq2_nlp.train as trainer
from moseq2_nlp.gridsearch import (find_gridsearch_results, generate_grid_search_worker_params,
                                   get_gridsearch_default_scans,
                                   wrap_command_with_local,
                                   wrap_command_with_slurm, write_jobs)
from moseq2_nlp.utils import (IntChoice, command_with_config, ensure_dir,
                              get_command_defaults, write_yaml)

from moseq2_nlp.data import get_raw_data, make_phrases_dataset
from moseq2_nlp.visualize import make_wordcloud
from tqdm import tqdm

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
@click.option('--model-path', type=click.Path(exists=True))
@click.option('--index-path', type=click.Path(exists=True))
@click.option('--representation', type=click.Choice(['embeddings', 'usages', 'transitions']), default='embeddings')
@click.option('--classifier', type=click.Choice(['logistic_regression', 'svm']), default='logistic_regression')
@click.option('--kernel', type=click.Choice(['linear', 'poly', 'rbf', 'sigmoid']), default='rbf')
@click.option('--emissions', is_flag=True)
@click.option('--custom-groupings', type=str, multiple=True, default=[])
@click.option('--num-syllables', type=int, default=70)
@click.option('--num-transitions', type=int, default=300)
@click.option('--min-count', type=int, default=1)
@click.option('--dm', default=2, type=IntChoice([0, 1, 2]))
@click.option('--embedding-dim', type=int, default=70)
@click.option('--embedding-window', type=int, default=4)
@click.option('--embedding-epochs', type=int, default=250)
@click.option('--bad-syllables', type=int, multiple=True, default=[-5])
@click.option('--scoring', type=str, default='accuracy')
@click.option('--k', type=int, default=1)
@click.option('--penalty', default='l2', type=click.Choice(['l1', 'l2', 'elasticnet']))
@click.option('--num-c', type=int, default=11)
@click.option('--seed', type=int, default=0)
@click.option('--config-file', type=click.Path())
def train(name, save_dir, model_path, index_path, representation, classifier, emissions, custom_groupings, num_syllables, num_transitions, min_count, dm, embedding_dim, embedding_window,
          embedding_epochs, bad_syllables, scoring, k, penalty, num_c, kernel, seed, config_file):

    trainer.train(name, save_dir, model_path, index_path, representation, classifier, emissions, custom_groupings, num_syllables, num_transitions, min_count, dm, embedding_dim, embedding_window,
          embedding_epochs, bad_syllables, scoring, k, penalty, num_c, kernel, seed)

@cli.command(name="generate-train-config", help="Generates a configuration file that holds editable options for training parameters.")
@click.option('--output-file', '-o', type=click.Path(), default='train-config.yaml')
def generate_train_config(output_file):
    output_file = os.path.abspath(output_file)
    write_yaml(output_file, get_command_defaults(train))
    print(f'Successfully generated train config file at "{output_file}".')

@cli.command(name="make-random-documents", help="Splits frames or emissions into sentences of random lengths and keeps track of which sentences belong to what class.")
@click.option('--model-path', type=str, default='/cs/usr/ricci/data/abraira/robust_septrans_model_20min_1000.p')
@click.option('--index-path', type=str, default='/cs/usr/ricci/data/abraira/moseq2-index.sex-genotype.20min.yaml')
@click.option('--splits',type=(float, float, float), default=(.4,.2,.4))
@click.option('--min-length', type=int, default=4)
@click.option('--max-length', type=int, default=32)
@click.option('--save-dir', type=str, default='/cs/usr/ricci/data/abraira/docs')
@click.option('--emissions', is_flag=True)
def make_random_documents(model_path, index_path, splits, min_length, max_length, save_dir, emissions):
    print(f'Gathering raw data for model "{model_path}".')
    sentences, out_groups = get_raw_data(model_path, index_path, emissions=emissions)

    fn_train = os.path.join(save_dir, 'ptb.train.txt')
    fn_val = os.path.join(save_dir, 'ptb.valid.txt')
    fn_test = os.path.join(save_dir, 'ptb.test.txt')
    
    labels_train = os.path.join(save_dir, 'train_labels.txt')
    labels_val = os.path.join(save_dir, 'valid_labels.txt')
    labels_test  = os.path.join(save_dir, 'test_labels.txt')

    print('Generating random documents.')
    with open(fn_train, 'w') as f1,  open(fn_val, 'w') as f2, open(fn_test, 'w') as f3, open(labels_train, 'w') as f4, open(labels_val, 'w') as f5, open(labels_test, 'w') as f6:
        for s, sentence in tqdm(enumerate(sentences)):
            counter = 0
            while counter < len(sentence):
                L = min(randint(min_length, high=max_length), len(sentence) - counter)
                snippet = ' '.join(sentence[counter:counter + L])
                counter += L
                which_doc = choice([f1, f2, f3],p=splits)

                if 'train' in which_doc.name:
                    which_lab = f4
                elif 'valid' in which_doc.name:
                    which_lab = f5
                else: 
                    which_lab = f6

                which_doc.write(snippet)
                which_doc.write('\n')
                which_lab.write(out_groups[s])
                which_lab.write('\n')
    
    print(f'Successfully generated random train/test documents at "{save_dir}".')

@cli.command(name='make-phrases', help='finds and saves compound modules')
@click.argument('model-path', type=click.Path(exists=True))
@click.argument('index-path', type=click.Path(exists=True))
@click.option('--save-path', type=click.Path(), default='./all_phrases.pickle')
@click.option('--threshes', type=float, multiple=True, default=[.01,.01])
@click.option('--n', type=int, default=2)
@click.option('--min-count', type=int, default=2)
@click.option('--visualize', is_flag=True)
@click.option('--wordcloud-path', type=click.Path(), default='.')
@click.option('--max-plot', type=int, default=15)
def make_phrases(model_path, index_path, save_path, wordcloud_path, threshes, n, min_count, visualize, max_plot):

    make_phrases_dataset(model_path, index_path, save_path, threshes, n, min_count)
    if visualize:
        print('Making word cloud')
        make_wordcloud(save_path, wordcloud_path, max_plot=max_plot)

@cli.command(name='grid-search', help='grid search hyperparameters')
@click.argument("scan_file", type=click.Path(exists=True))
@click.option('--env', type=str, default='moseq2-nlp', help="Environment in which to run jobs")
@click.option('--save-dir', type=click.Path(), default=os.path.join(os.getcwd(), 'worker-configs'), help="Directory to save worker configurations")
@click.option('--cluster-type', default='local', type=click.Choice(['local', 'slurm']))
@click.option('--slurm-ncpus', type=int, default=1, help="Number of CPUs per job. Only for SLURM")
@click.option('--slurm-memory', type=str, default="2GB", help="Amount of memory per job. Only for SLURM")
@click.option('--slurm-wall-time', type=str, default='6:00:00', help="Max wall time per job. Only for SLURM")
@click.option('--slurm-killable', is_flag=True)
def grid_search(scan_file, env, save_dir, cluster_type, slurm_ncpus, slurm_memory, slurm_wall_time, slurm_killable):

    worker_dicts = generate_grid_search_worker_params(scan_file)

    if cluster_type == 'local':
        cluster_wrap = wrap_command_with_local
    elif cluster_type == 'slurm':
        cluster_wrap = partial(wrap_command_with_slurm, env=env, ncpus=slurm_ncpus, memory=slurm_memory, wall_time=slurm_wall_time, killable=slurm_killable)
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

if __name__ == '__main__':
    cli()
