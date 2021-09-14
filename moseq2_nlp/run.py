import itertools
import os
import sys
from typing import Callable, List, Literal

import numpy as np

from moseq2_nlp.utils import read_yaml, write_yaml


def get_scan_values(scale: Literal['log', 'linear', 'list'], range: List, type='float') -> List:
    ''' Generate concrete scan values given a scan specification

    Parameters:
        scale (Literal['log', 'linear', 'list']): type of scale to scan.
        range (List): arguments for the scale
        type (dtype-like): dtype for the scale

    Returns:
        (List): List of values to scan over
    '''
    if scale == 'list':
        return np.array(range).astype(type).tolist()
    elif scale == 'linear':
        return np.linspace(*range).astype(type).tolist()
    elif scale == 'log':
        ls_args = [np.log10(x) if i < 2 else x for i, x in enumerate(range)]
        return np.logspace(*ls_args).astype(type).tolist()
    else:
        raise ValueError(f'Unknown scale {scale}')


def wrap_command_with_slurm(cmd: str, partition: str, ncpus: int, memory: str, wall_time: str) -> str:
    ''' Wraps a command to be run as a SLURM sbatch job

    Parameters:
        cmd (str): Command to be wrapped
        partition (str): Partition on which to run this job
        ncpus (int): Number of CPU cores to allocate to this job
        memory (str): Amount of memory to allocate to this job. ex: "2GB"
        wall_time (str): Amount of wall time allocated to this job. ex: "1:00:00"

    Returns:
        (str): the slurm wrapped command
    '''
    preamble = f'sbatch --partition {partition} --nodes 1 --ntasks-per-node 1 --cpus-per-task={ncpus} --mem {memory} --time {wall_time}'
    escaped_cmd = cmd.replace('"', r'\"')
    return f'{preamble} --wrap "{escaped_cmd}";'


def wrap_command_with_local(cmd: str) -> str:
    ''' Wraps a command to be run locally. Admittedly, this does not do too much

    Parameters:
        cmd (str): Command to be wrapped

    Returns:
        (str): the wrapped command
    '''
    return cmd


def generate_grid_search_worker_params(scan_file: str) -> List[dict]:
    ''' Given a path to YAML scan configuration file, read the contents
        and generate a dictionary for each implied job

    Parameters:
        scan_file (str): path to a yaml scan configuration file

    Returns:
        (List[dict]): a list of dicts, each dict containing parameters to a single job
    '''
    scan_settings = read_yaml(scan_file)

    base_parameters = scan_settings['parameters']
    scans = scan_settings['scans']

    worker_dicts = []
    for scan in scans:

        scan_param_products = []
        if 'scan' in scan:
            scan_param_gens = {}
            for sp in scan['scan']:
                scan_param_gens[sp['parameter']] = get_scan_values(sp['scale'], sp['range'], sp['type'])

            for params in itertools.product(*scan_param_gens.values()):
                scan_param_products.append(dict(zip(scan_param_gens.keys(), params)))
        else:
            # single empty dict, since we want to run with the scan parameters once
            # even though we dont have any iterable parameters
            scan_param_products.append({})

        if 'parameters' in scan:
            scan_base_params = scan['parameters']
        else:
            scan_base_params = {}


        for spp in scan_param_products:
            # scan-specific params, combo of static and dynamic
            scan_params = {**scan_base_params, **spp}
            # final params incorporating scan-specific params
            final_params = {**base_parameters, **scan_params}

            # rename the job, based on the specific params
            name_suffix = '_'.join([f'{k}-{v}' for k, v in scan_params.items()])
            final_params['name'] = f"{final_params['name']}_{name_suffix}"

            worker_dicts.append(final_params)

    return worker_dicts


def write_jobs(worker_dicts: List[dict], cluster_format: Callable[[str],str], dest_dir: str) -> None:
    ''' Write job configurations to YAML files, and write job invocations to stdout

    Parameters:
        worker_dicts (List[dict]): Job configurations to write
        cluster_format (Callable[[str], str]): A callable to format the job invoation for a given environment
        dest_dir (str): directory to write job configurations to
    '''
    for worker in worker_dicts:
        worker_dest = os.path.join(dest_dir, f"{worker['name']}.yaml")
        write_yaml(worker_dest, worker)

        work_cmd = f'moseq2-nlp train --config "{worker_dest}";'
        full_cmd = cluster_format(work_cmd)

        sys.stdout.write(full_cmd+'\n')


def get_gridsearch_default_scans() -> List:
    ''' Generate default scan configuration
    '''
    return [
    {
        'parameters': {
            'representation': 'usages'
        }
    }, {
        'parameters': {
            'representation': 'transitions'
        },
        'scan': [
            {
                'parameter': 'num_transitions',
                'type': 'int',
                'scale': 'log',
                'range': [70, 4900, 8]
            }
        ]
    }, {
        'parameters': {
            'representation': 'embeddings'
        },
        'scan': [
            {
                'parameter': 'emissions',
                'type': 'bool',
                'scale': 'list',
                'range': [True, False]
            }, {
                'parameter': 'embedding_window',
                'type': 'int',
                'scale': 'list',
                'range': [2, 4, 8, 16, 32, 64]
            }, {
                'parameter': 'embedding_dim',
                'type': 'int',
                'scale': 'log',
                'range': [70, 4900, 8]
            }, {
                'parameter': 'embedding_epochs',
                'type': 'int',
                'scale': 'list',
                'range': [50, 100, 150, 200, 250]
            }
        ]
    }]
