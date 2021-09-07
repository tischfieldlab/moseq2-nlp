# moseq_nlp
Interrogating Moseq data using a NLP-based approach

## Basic usage
To train a classifier on MoSeq syllables, run `python train.py --experiment <my_experiment>`. `<my_experiment>` should be the name of a directory in your `data_dir` and should be included in the `set_data` function in `utils.py`.  E.g. 

`python train.py --experiment 2021-02-19_Meloxicam`

## Arguments
All arguments can be used in the command line (e.g. `python train.py --myarg <arg>`) or by adjusting the relevant key in `config.cfg`.

* `seed` (int) : Random seed for numpy. Default 0.
* `name` (str) : Custom name for your experiment. Will be appended to `experiment` to form the directory name for saving. If None, the save directory will just be `experiment`.
* `data_dir` (str): The directory where your data folder (with `.p` and `.yaml` files) is stored.
* `save_dir` (str): The directory where you want to save your results folder. 
* `experiment` (str): The name of the experiment. Should be a folder in `data_dir` and should be contained in the `set_data` function in `utils.py`.
* `timepoint` (int): Only used for temporally-distinct data spread across different .p files.
* `representation` (str). Which type of features to use for the classifier. Can be `usages`, `transitions` or `embeddings`.
* `emissions` (bool): If True, uses emissions. If False, uses raw frame data. 
* `custom_labels` (list of ints): If you want to group your data differently, set their labels here. For instance, if you have five experimental groups, but you want the first three to be in the same class and the second two to be in another class, you can set this to be `[0,0,0,1,1]`. Otherwise, just leave as an increasing list like [0,1,2,3,4].
* `custom_label_names` (list of str) : Names you want to give your custom labels. List should be as long as there are unique labels (e.g. for the above example, you could use `[group_A, `group_B`]`)
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
