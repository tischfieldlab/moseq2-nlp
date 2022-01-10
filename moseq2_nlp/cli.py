import os
import sys
from distutils import util
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

from moseq2_nlp.data import get_raw_data
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
@click.option('--emissions', is_flag=True, type=lambda x:bool(util.strtobool(x)))
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
@click.option('--model_path', type=str, default='/cs/usr/ricci/data/abraira/robust_septrans_model_20min_1000.p')
@click.option('--index_path', type=str, default='/cs/usr/ricci/data/abraira/moseq2-index.sex-genotype.20min.yaml')
@click.option('--splits',type=str,multiple=True,default=[.4,.2,.4])
@click.option('--min_length', type=int, default=4)
@click.option('--max_length', type=int, default=32)
@click.option('--save_dir', type=str, default='/cs/usr/ricci/data/abraira/docs')
@click.option('--emissions', '-e',type=lambda x:bool(util.strtobool(x)))
def make_random_documents(model_path, index_path, splits, min_length, max_length, save_dir, emissions):
    print(f'Gathering raw data for model "{model_path}".')
    sentences, out_groups = get_raw_data(model_path, index_path, emissions=emissions)

    splits = [float(s) for s in splits]

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

@cli.command(name='grid-search', help='grid search hyperparameters')
@click.argument("scan_file", type=click.Path(exists=True))
@click.option('--save-dir', type=click.Path(), default=os.path.join(os.getcwd(), 'worker-configs'), help="Directory to save worker configurations")
@click.option('--cluster-type', default='local', type=click.Choice(['local', 'slurm']))
@click.option('--slurm-partition', type=str, default='main', help="Partition on which to run jobs. Only for SLURM")
@click.option('--slurm-ncpus', type=int, default=1, help="Number of CPUs per job. Only for SLURM")
@click.option('--slurm-memory', type=str, default="2GB", help="Amount of memory per job. Only for SLURM")
@click.option('--slurm-wall-time', type=str, default='6:00:00', help="Max wall time per job. Only for SLURM")
def grid_search(scan_file, save_dir, cluster_type, slurm_partition, slurm_ncpus, slurm_memory, slurm_wall_time):

    worker_dicts = generate_grid_search_worker_params(scan_file)

    if cluster_type == 'local':
        cluster_wrap = wrap_command_with_local
    elif cluster_type == 'slurm':
        cluster_wrap = partial(wrap_command_with_slurm, partition=slurm_partition, ncpus=slurm_ncpus, memory=slurm_memory, wall_time=slurm_wall_time)
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
@click.argument("results_directory", type=click.Path(exists=True))
def aggregate_gridsearch_results(results_directory):
    
    
    results = find_gridsearch_results(results_directory)
    results.to_csv(os.path.join(results_directory, 'gridsearch-aggregate-results.tsv'), sep='\t', index=False)





if __name__ == '__main__':
    cli()
