# moseq_nlp
Interrogating Moseq data using a NLP-based approach

## Basic usage
To train a classifier on MoSeq syllables, run `python train.py --experiment <my_experiment>`. `<my_experiment>` should be the name of a directory in your `data_dir` and should be included in the `set_data` function in `utils.py`.  E.g. 

`python train.py --experiment 2021-02-19_Meloxicam`

## Arguments
All arguments can be used in the command line (e.g. `python train.py --myarg <arg>`) or by adjusting the relevant key in `config.cfg`.

* `seed` (int) : Random seed for numpy. Default 0.
* `name` (str) : Name of directory where results will be saved inside `save_dir`.
* `model_path` (str): Full path to the Moseq `.p` file.
* `index_path` (str): Full path to the Moseq `.yaml` file.
* `save_dir` (str): The directory where you want to save your results folder. 
* `representation` (str). Which type of features to use for the classifier. Can be `usages`, `transitions` or `embeddings`.
* `emissions` (bool): If True, uses emissions. If False, uses raw frame data. 
* `custom_groupings` (list of strings) : To collect several classes `'a', 'b', ...,'z'` into a single group add an elment to this list of the form `'a,b,...,z'` where sub-class names are separated by commas. For instance, if you have four-class data (`'a', 'b', 'c', 'd'`) and you want to group the first three into one class, set `custom_groupings` to `['a,b,c','d']`. If you want to use the raw classes, leave as an empty list.  
* `num_syllables` (int) : How many types of syllables to use (max 100).
* `num_transitions` (int) : How many transitions to use, counting from the the most frequent (max num_syllables^2)
* `min_count` (int) : Minimum number of occurrences of a syllable in the raw data (emissions or frames) for it to be included in the embedding learning process.
* `dm` (int) : If 1 uses the distributed memory Doc2Vec representation. If 0, uses CBOW. If 2, uses an average of the two. 
* `embedding_dim` (int) : Dimension of the embedding space.
* `embedding_window` (int) : Size of the context window to use during Doc2Vec training.
* `embedding_epochs` (int) : How many full passes to go through the data during Doc2Vec training.
* `bad_syllables` (list of ints) : Which sylllabes to exclude from training. 
* `scoring` (str) : Which score to use for the classifier. Should be recognized by sklearn.metrics.
* `K` (int) : The sizes of the held-out set for K-fold cross-validation. E.g, for leave-one-out, set `K=1`.
* `penalty` (str) : Type of penalty used by logistic regressor. Should be recognized by this function. 
* `num_C` (int) : How many regularization terms to search over, logarithmically spaced between 1e-5 and 1e5. 
