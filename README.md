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
moseq2-nlp generate-config --output-file config.yaml
```

Edit the configuration in the yaml file. Any arguments/options supplied on the command line will override those in the config file.


To train a classifier on MoSeq syllables, run:
```
moseq2-nlp train --config-file <path_to_config_file>
```

`moseq2-nlp train --config-file ./config.yaml`

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
