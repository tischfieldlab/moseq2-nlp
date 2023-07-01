import glob
import itertools
import os
import sys
import uuid
from typing import Any, Callable, List, Literal, Optional, Protocol

import numpy as np
import pandas as pd

from moseq2_nlp.util import ensure_dir, read_yaml, write_yaml
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
import pdb


def get_scan_values(scale: Literal["log", "linear", "list"], range: List, type="float") -> List:
    """Generate concrete scan values given a scan specification.

    Args:
        scale: (Literal['log', 'linear', 'list']) type of scale to scan.
        range: (List) arguments for the scale
        type: (dtype-like) dtype for the scale

    Returns:
        (List): List of values to scan over
    """
    if scale == "list":
        return np.array(range).astype(type).tolist()
    elif scale == "linear":
        return np.linspace(*range).astype(type).tolist()
    elif scale == "log":
        ls_args = [np.log10(x) if i < 2 else x for i, x in enumerate(range)]
        return np.logspace(*ls_args).astype(type).tolist()
    else:
        raise ValueError(f"Unknown scale {scale}")


class CommandWrapper(Protocol):
    """Wraps command."""

    def __call__(self, cmd: str, output: Optional[str] = None, **kwds: Any) -> str:
        """Calls command.

        Args:
            cmd: string carrying out command
            output: output option
        """
        ...


def wrap_command_with_slurm(
    cmd: str, preamble: str, partition: str, ncpus: int, memory: str, wall_time: str, extra_params: str, output: Optional[str] = None
) -> str:
    """Wraps a command to be run as a SLURM sbatch job.

    Args:
        cmd (str): Command to be wrapped
        preamble (str): Commands to be run prior to `cmd` as part of this job
        partition (str): Partition on which to run this job
        ncpus (int): Number of CPU cores to allocate to this job
        memory (str): Amount of memory to allocate to this job. ex: "2GB"
        wall_time (str): Amount of wall time allocated to this job. ex: "1:00:00"
        extra_params (str): Extra parameters to pass to slurm sbatch command
        output (str): Path of file to write output to

    Returns:
        (str): the slurm wrapped command
    """
    # setup basic parameters for slurm's `sbatch` command:
    #   important to set --nodes to 1 and --ntasks-per-node to one 1 or
    #   the multiple --cpus-per-task may be split over multiple nodes!
    sbatch_cmd = f"sbatch --partition {partition} --nodes 1 --ntasks-per-node 1 --cpus-per-task {ncpus} --mem {memory} --time {wall_time}"

    # if the user requests job log output to a file, set that up
    if output is not None:
        sbatch_cmd += f' --output "{output}"'

    # if any extra params for slurm, add them
    if len(extra_params) > 0:
        sbatch_cmd += f" {extra_params}"

    if len(preamble) > 0:
        # if preamble does not end with semicolon, add one to separate from main command
        if not preamble.endswith(";"):
            preamble = preamble + "; "

        # ensure there is a space separating preamble from main command
        if not preamble.endswith(" "):
            preamble = preamble + " "

        # escape any quotes within the preamble
        preamble = preamble.replace('"', r"\"")

    # escape any quotes in the command
    escaped_cmd = cmd.replace('"', r"\"")

    # put it all togher and return the final wrapped command
    return f'{sbatch_cmd} --wrap "{preamble}{escaped_cmd}";'


def wrap_command_with_local(cmd: str, output: Optional[str] = None) -> str:
    """Wraps a command to be run locally. Admittedly, this does not do too much.

    Args:
        cmd (str): Command to be wrapped
        output (str): Path of file to write output to

    Returns:
        (str): the wrapped command
    """
    if output is not None:
        return cmd
    else:
        return cmd + f' > "{output}"'


def generate_grid_search_worker_params(scan_file: str) -> List[dict]:
    """Given a path to YAML scan configuration file, read the contents and generate a dictionary for each implied job.

    Args:
        scan_file (str): path to a yaml scan configuration file

    Returns:
        (List[dict]): a list of dicts, each dict containing parameters to a single job
    """
    scan_settings = read_yaml(scan_file)

    base_parameters = scan_settings["parameters"]
    scans = scan_settings["scans"]

    worker_dicts = []
    for scan in scans:
        scan_param_products = []
        if "scan" in scan:
            scan_param_gens = {}
            for sp in scan["scan"]:
                scan_param_gens[sp["parameter"]] = get_scan_values(sp["scale"], sp["range"], sp["type"])

            for params in itertools.product(*scan_param_gens.values()):
                scan_param_products.append(dict(zip(scan_param_gens.keys(), params)))
        else:
            # single empty dict, since we want to run with the scan parameters once
            # even though we dont have any iterable parameters
            scan_param_products.append({})

        if "parameters" in scan:
            scan_base_params = scan["parameters"]
        else:
            scan_base_params = {}

        for spp in scan_param_products:
            # scan-specific params, combo of static and dynamic
            scan_params = {**scan_base_params, **spp}
            # final params incorporating scan-specific params
            final_params = {**base_parameters, **scan_params}

            # rename the job, based on the specific params
            name_suffix = "_".join([f"{k}-{v}" for k, v in scan_params.items()])
            final_name = f"{final_params['name']}_{name_suffix}"
            final_name = final_name.replace('/',"_")
            final_params["name"] = final_name

            worker_dicts.append(final_params)

    return worker_dicts


def write_jobs(worker_dicts: List[dict], cluster_format: CommandWrapper, dest_dir: str) -> None:
    """Write job configurations to YAML files, and write job invocations to stdout.

    Args:
        worker_dicts (List[dict]): Job configurations to write
        cluster_format (Callable[[str], str]): A callable to format the job invoation for a given environment
        dest_dir (str): directory to write job configurations to
    """
    for worker in worker_dicts:
        worker_dest = os.path.join(dest_dir, f"{worker['name']}.yaml")
        write_yaml(worker_dest, worker)

        ensure_dir(worker["save_dir"])
        output = os.path.join(worker["save_dir"], f"{worker['name']}.log")

        work_cmd = f'moseq2-nlp train --config-file "{worker_dest}";'
        full_cmd = cluster_format(work_cmd, output=output)

        sys.stdout.write(full_cmd + "\n")


def get_gridsearch_default_scans() -> List:
    """Generate default scan configuration."""
    return [
        {"parameters": {"representation": "usages"}},
        {
            "parameters": {"representation": "transitions"},
            "scan": [{"parameter": "num_transitions", "type": "int", "scale": "linear", "range": [10, 300, 10]}],
        },
        {
            "parameters": {"representation": "embeddings"},
            "scan": [
                {"parameter": "emissions", "type": "bool", "scale": "list", "range": [True, False]},
                {"parameter": "embedding_window", "type": "int", "scale": "list", "range": [2, 4, 8, 16, 32, 64]},
                {"parameter": "embedding_dim", "type": "int", "scale": "linear", "range": [10, 300, 10]},
                {"parameter": "embedding_epochs", "type": "int", "scale": "list", "range": [50, 100, 150, 200, 250]},
            ],
        },
    ]

def parse_exp_name(path: str) -> dict:
    """Parse exp yaml into parameters

    Args:
        path (str): yaml path
    """
    exp_dict = {}
    path_copy = path

    #base_name
    rep_idx = path_copy.index('representation')
    base_name = path_copy[0:rep_index]
    path_copy = path_copy[rep_index:]

    #rep_name
    rep_end = path_copy.find('_')
    rep_start = path_copy[15]
    rep_name = path_copy[rep_start:rep_end]
    path_copy = path_copy[rep_end:]

    #preprocessing_name
    emission_start = path_copy.find('emissions')
    preprocessing_name = path_copy[:emission_start]
    path_copy = path_copy[emission_start:]

    #emissions
    emissions_end = path_copy.find('_')
    emissions_start = path_copy.find('-') + 1
    emissions = path_copy[emissions_start:emissions_end]
    path_copy = path_copy[emissions_end:]

    return NotImplemented

def find_gridsearch_results(path: str) -> pd.DataFrame:
    """Find and aggregate grid search results.

    Args:
        path (str): path to search for experiments
    """
    experiments = glob.glob(os.path.join(path, "*", "results.yaml"))

    exp_data = []
    for exp in experiments:
        id = uuid.uuid4()
        data = read_yaml(exp)

        # tag each dict with the model ID
        time_data = {f"time_{k}": v for (k, v) in data["compute_times"].items()}
        train_results = {f'train_{key}': val for (key,val) in data['model_performance_train']['classification_report']['weighted avg'].items()}
        test_results = {f'test_{key}': val for (key,val) in data['model_performance_test']['classification_report']['weighted avg'].items()}

        exp_data.append({'name': exp, **time_data, **train_results, **test_results})

    return pd.DataFrame(exp_data)


def get_best_model(path: str, key="best_accuracy"):
    """Finds and returns the best model according to a particular measure.

    Args:
        path (str): path to search for experiments
        key (sr): measure by which to rank experiments
    """
    df = find_gridsearch_results(path).sort_values(key, ascending=False)
    return df.iloc[0]


def observe_gs_variation(path: str, representation_name: str, dep_var_name: str, ind_var_name: str):
    """Plots how two gridsearch variables covary.

    Args:
        path: str, where gridsearch results are saved in the form of a pandas dataframe
        : str indicating which features among usages, transitions and embeddings were used
        dep_var_name: str, first variable name, plotted on y axis
        ind_var_name: str, second variable name, plotted on x axis
    """
    df = find_gridsearch_results(path)
    current_df = df.loc[df["representation"] == representation_name]
    dep_var = current_df[dep_var_name].values
    ind_var = current_df[ind_var_name]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(ind_var, dep_var)
    ax.set_title(f"{ind_var_name}, vs {dep_var_name}. Rep: {representation_name}")
    ax.set_xlabel(ind_var_name)
    ax.set_ylabel(dep_var_name)
    plt.show()
    plt.close()

def train_params_to_features(train_params):

    feature_dims = ['data_path','dm', 'embedding_dim', 'embedding_epochs', 'embedding_window', 'emissions', 'num_syllables', 'num_transitions', 'representation']

    feature_vector = []

    for dim in feature_dims:
        feature = train_params[dim]
        
        if dim == 'data_path':
            splt = feature.split('_')
            res_plus = splt[-1]
            res = res_plus.split('.')[0]
            num_feature = int(res)
        elif dim == 'emissions':
            num_feature = 1 * feature
        elif dim == 'embedding_dim':
            num_feature = np.log2(feature)
        elif dim == 'representation':
            if feature == 'usages':
                num_feature = 0
            elif feature == 'transitions':
                num_feature = 1
            else:
                num_feature = 2
        else: 
            num_feature = feature
        feature_vector.append(num_feature)

    return np.array(feature_vector), feature_dims

def get_gridsearch_xy(results_yaml, params_yaml, dep_var='test_f1'):
    if dep_var == 'test_f1':
        y = results_yaml['model_performance_test']['classification_report']['weighted avg']['f1-score']
    elif dep_var == 'train_f1':
        y = results_yaml['model_performance_train']['classification_report']['weighted avg']['f1-score']
    elif dep_var == 'classifier_time':
        y = results_yaml['Compute_times']['Classifier']
    elif dep_var == 'features_time':
        y = results_yaml['Compute_times']['Features']
    elif dep_var == 'data_time':
        y = results_yaml['Compute_times']['Data']
    else:
        raise ValueError('Dependent variable not recognized!')
    x, _ = train_params_to_features(params_yaml)

    return x, y

def gridsearch_regressor(results_dir, dep_var='test_f1'):
    
    all_results = glob.glob(os.path.join(results_dir, "*", "results.yaml"))
    all_params = glob.glob(os.path.join(results_dir, "*", "train_params.yaml"))

    _, param_names = train_params_to_features(read_yaml(all_params[0]))

    X = []
    Y = []
    for results, params in zip(all_results, all_params):
        results_yaml = read_yaml(results)
        params_yaml = read_yaml(params)
        x, y = get_gridsearch_xy(results_yaml, params_yaml, dep_var=dep_var)
        X.append(x)
        Y.append(y)
    X = np.array(X)
    X = (X/X.max(0))
    Y = np.array(Y)

    #clf = Lasso(alpha=1e-4)
    clf = LinearRegression()
    print('Training linear regressor')
    clf.fit(X,Y)

    coeff_dict = {param:coeff for param, coeff in zip(param_names, clf.coef_)}
    print(f'r2 = {clf.score(X,Y)}')
    print(coeff_dict)
