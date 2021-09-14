# moseq2-nlp
Interrogating moseq data using a NLP-based approach

## Install

Create a new conda virtual environment:
```
conda create -n moseq2-nlp python=3.8
```

Activate the newly created environment:
```
conda activate moseq2-nlp
```

Install moseq2-viz dependency:
```
pip install git+https://github.com/tischfieldlab/moseq2-viz.git     # if you like to use git over https
pip install git+ssh://git@github.com/tischfieldlab/moseq2-viz.git   # if you like to use git over ssh
```

Install this repo:
```
pip install git+https://github.com/tischfieldlab/moseq2-nlp.git     # if you like to use git over https
pip install git+ssh://git@github.com/tischfieldlab/moseq2-nlp.git   # if you like to use git over ssh
```

## Basic usage
Options for training can be specified via a yaml configuration file. To generate a template configuration file, execute the following command:
```
moseq2-nlp generate-train-config --output-file ./train-config.yaml
```

Edit the configuration in the yaml file. Any arguments/options supplied on the command line will override those in the config file.


To train a classifier on MoSeq data, run:
```
moseq2-nlp train --config-file ./train-config.yaml
```


## Arguments
All arguments can be used in the command line (e.g. `moseq2-nlp train --myarg <arg>`) or by adjusting the relevant key in `config.yaml`.

* `seed` (int): Random seed for numpy. Default 0.
* `name` (str): Name of directory where results will be saved inside `save_dir`.
* `model-path` (str): Full path to the Moseq `.p` file.
* `index-path` (str): Full path to the Moseq `.yaml` file.
* `save-dir` (str): The directory where you want to save your results folder.
* `representation` (str): Which type of features to use for the classifier. Can be `usages`, `transitions` or `embeddings`.
* `emissions` (bool): If True, uses emissions. If False, uses raw frame data.
* `custom-groupings` (list of strings) : To collect several classes `'a', 'b', ...,'z'` into a single group add an elment to this list of the form `'a,b,...,z'` where sub-class names are separated by commas. For instance, if you have four-class data (`'a', 'b', 'c', 'd'`) and you want to group the first three into one class, set `custom-groupings` to `['a,b,c','d']`. If you want to use the raw classes, leave as an empty list.
* `num-syllables` (int): How many types of syllables to use (max 100).
* `num-transitions` (int): How many transitions to use, counting from the the most frequent (max num_syllables^2)
* `min-count` (int): Minimum number of occurrences of a syllable in the raw data (emissions or frames) for it to be included in the embedding learning process.
* `dm` (int): If 1 uses the distributed memory Doc2Vec representation. If 0, uses CBOW. If 2, uses an average of the two.
* `embedding-dim` (int): Dimension of the embedding space.
* `embedding-window` (int): Size of the context window to use during Doc2Vec training.
* `embedding-epochs` (int): How many full passes to go through the data during Doc2Vec training.
* `bad-syllables` (list of ints): Which sylllabes to exclude from training.
* `scoring` (str): Which score to use for the classifier. Should be recognized by sklearn.metrics.
* `K` (int): The sizes of the held-out set for K-fold cross-validation. E.g, for leave-one-out, set `K=1`.
* `penalty` (str): Type of penalty used by logistic regressor. Should be recognized by this function.
* `num-C` (int): How many regularization terms to search over, logarithmically spaced between 1e-5 and 1e5.
* `config-file` (str): Path to a configuration file to read containing any of the above arguments.


## Running Parameter Scans
It is often useful to run hyperparameter searches, à la grid-search.

### Generate Grid-Search Configuration
Options for grid search can be specified via a yaml configuration file. To generate a template configuration file for grid-search, execute the following command:
```
moseq2-nlp generate-gridsearch-config --output-file ./gridsearch-config.yaml
```

### Edit Grid-Search Configuration
There are a few important sections of the grid-search configuration. First are the base parameters, specified by the root key `parameters`. These values are the same as the train configuration keys, and provide default values for all parameters.

The next important section of the grid-search configuration is the root key `scans`. The value of this section sould be a list of scan configurations. Each scan configuration consists of two possible keys:
- `parameters`: dictionary of train parameters which will override the base parameters.
- `scan`: list of parameter scan configurations.

Each scan configuration must include the following keys:
- `parameter`: name of the parameter to scan over
- `type`: data type of the parameter value (e.x. `bool`, `int`, or `float`)
- `scale`: type of scale to use to generate values. Options include:
    - `list`: `range` will be interpreted as a literal list of values to be used
    - `linear`: `range` will be interpreted as the arguments to numpy.linspace
    - `log`: `range` will be interpreted as the arguments to numpy.logspace

For each scan configuration, the cartesian product of scan values is taken and combined with scan-specific parameters and base parameters.

### Generate Grid-Search Jobs
Once you have edited the grid-search configuration to your liking, you may now generate the files and commands necessary to execute these jobs:
```
moseq2-nlp grid-search --save-dir ./job-configs ./gridsearch-config.yaml > ./jobs.sh
```
This will save one yaml file per job generated into the directory specified by the parameter `--save-dir`, and will output the job invocation commands to `stdout`. Given the example above, the following would execute the generated jobs:
```
chmod +x ./jobs.sh
./jobs
```

### Cluster Support for Grid-Search
We provide some basic support for generating the job commands for use on computer cluster. The default is to output commands for running locally.

To generate commands sutible for submission to a SLURM cluster, utilize the option `--cluster-type slurm` and the related `--slurm-*` options to specify slurm scheduler options.
