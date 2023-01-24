import os
import time
from typing import List, Literal, Union

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from moseq2_nlp.data import get_embedding_representation, get_transition_representation, get_usage_representation, load_groups, get_emissions
from moseq2_nlp.util import ensure_dir, write_yaml, get_unique_list_elements
import pickle
import pandas as pd
import pdb

import warnings
warnings.filterwarnings("ignore")

Representation = Literal['embeddings', 'usages', 'transitions']
Classifier = Literal['logistic_regression', 'svm']
Penalty = Literal['l1', 'l2', 'elasticnet']

def train(name: str, save_dir: str, data_path: str, representation: Representation, classifier: Classifier, emissions: bool,
        num_syllables: int, num_transitions: int, min_count: int, negative: int, dm: Literal[0,1,2], embedding_dim: int, embedding_window: int,
          embedding_epochs: int, bad_syllables: List[int], test_size: float, K: int, penalty: Penalty, num_c: int, multi_class: str, kernel: str, seed:int, split_seed:int=None, verbose:int=0):

    """Compute animal features and train classifier. .

    Args:
        name: string which names the experiment  
        save_dir: directory in which the features and classifier results will be saved
        representation: string indicating which representation, among `usages`, `transitions` and `embeddings`, will be used
        classifier: string, either `logistic_regression` or `svm`, which determines the classifier type
        emissions: boolean determining whether behavior will be represented as frames (False) or emisisons (True)
        num_syllables: int indicating the total number of syllables to include in the analysis. Syllables with higher values are excluded. 
        num_transitions: int indicating the total number of transitions to include in the `transitions` representation, should that representation be chosen
        min_count: int controling minimum number of times a syllable must appear for it to be included in the `embedding` representation, should that representation be chosen
        negative: int which is used as the exponent for negative sampling in doc2vec
        dm: int, either 0, 1, or 2, controling which among dbow (0), distributed memory (1), or an average of the two (2), should be used for the `embedding` representation
        embedding_dim: int, dimension of the doc2vec embedding space
        embedding_window: int, size of context window for doc2vec embeddings
        embedding_epochs: int, number of passes through the data set during doc2vec training
        bad_syllables: list of ints indicating which syllables should be excluded form the analysis
        test_size: float between 0 and 1 indicating proportion of data set to be held out for testing
        K: int, number of splits for cross validation
        penalty: string controling which type of penalty to use for the classifier (see svm and logistic regressor docs in sklearn)
        num_c: int, how many regularizers to gridsearch over in cross validation, logarithmically spaced between [1e-5 and 1e5]
        multi_class: string determining which multiclass scheme to use for the logistic regressor (see sklearn docs)
        kernel:  string determining which type of `svm` kernel should be used (see sklearn docs)
        seed: int, random seed for feature learning
        split_seed: int, random seed for the train-test split
        verbose: int controling verbosity of sklearn classifiers

    See also: `train_regressor, train_svm`
    """

    np.random.seed(seed)    
    save_dict = {}
    times = {'Preamble': 0.0, 'Data': 0.0, 'Features': 0.0, 'Classifier': 0.0}

    start = time.time()
    bad_syllables = [int(bs) for bs in bad_syllables]
    exp_dir = ensure_dir(os.path.join(save_dir, name))

    times['Preamble'] = time.time() - start

    # Load data
    with open(os.path.join(data_path,'sentences.pkl'),'rb') as fn:
        sentences = pickle.load(fn)
        if emissions: sentences = get_emissions(sentences)

    with open(os.path.join(data_path,'labels.pkl'),'rb') as fn:
        labels = pickle.load(fn)

    print('Getting features')
    if representation == 'embeddings':
        features = get_embedding_representation(sentences, emissions=emissions, bad_syllables=bad_syllables,
                            dm=dm, embedding_dim=embedding_dim, embedding_window=embedding_window, embedding_epochs=embedding_epochs, min_count=min_count, negative=negative,
                            model_dest=os.path.join(exp_dir, 'doc2vec'), ablation='none', phrase_path=None, seed=seed)

    elif representation == 'usages':
        features = get_usage_representation(sentences, num_syllables)

    elif representation == 'transitions':
        features = get_transition_representation(sentences, num_transitions, max_syllable=num_syllables)
    else:
        raise ValueError('Representation type not recognized. Valid values are "usages", "transitions" and "embeddings".')

    unique_labels = get_unique_list_elements(labels)

    # Make train/test splits
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=split_seed, stratify=labels)

    times['Features'] = time.time() - start

    start = time.time()
    print('Training classifier')
    if classifier == 'logistic_regressor':
        # Train logistic regressor and CV over C using validation data from the training set.
        clf = train_regressor(X_train, y_train, K, penalty, num_c, seed, multi_class, verbose=verbose)
    elif classifier == 'svm':
        clf = train_svm(X_train, y_train, K, penalty, num_c, seed, verbose=verbose)
    else:
        raise ValueError(f'Classifier {classifier} not recognized')

    y_pred_train = clf.predict(X_train)
    report_train = classification_report(y_train, y_pred_train, output_dict=True)

    y_pred_test = clf.predict(X_test)
    report_test = classification_report(y_test, y_pred_test, output_dict=True)

    cm_train = confusion_matrix(y_train, y_pred_train, labels=unique_labels)
    cm_test = confusion_matrix(y_test, y_pred_test, labels=unique_labels)
    
    times['Classifier'] = time.time() - start

    save_dict['model_performance_train'] = {f'classification_report': report_train}
    save_dict['model_performance_test']  = {f'classification_report': report_test}

    save_dict['compute_times'] = times
    write_yaml(os.path.join(exp_dir, 'results.yaml'), save_dict)

    for nm, lb_true, lb_pred in zip(['train', 'test'], [y_train, y_test], [y_pred_train, y_pred_test]):
        cm = confusion_matrix(lb_true, lb_pred, labels=unique_labels, normalize='true')
        df = pd.DataFrame(cm)
        df.set_axis([str(ul) + '_pred' for ul in unique_labels], axis=0, inplace=True)
        df.set_axis([str(ul) + '_true' for ul in unique_labels], axis=1, inplace=True)
        df
        df.to_pickle(os.path.join(exp_dir, f'cm_{nm}.pkl'))

def train_regressor(features, labels, K: int, penalty: Penalty, num_c: int, seed: int, multi_class: Literal['auto', 'multi_class', 'ovr'], verbose: int=0):

    """ Trains a Kfold cross-validated logistic regressor and returns fitted classifier

        Args:
            features: a sample x feature floating point numpy array of labeled data to be classified
            labels: an iterable of integer labels for the samples
            K: integer number of splits for cross validation
            penalty: literal indicating which sort of regularization to use, `l1`, `l2` or `elasticnet``
            num_c: integer number of regularizer constants to search over, logarithmically spaced between 1e-5 and 1e5
            seed: integer random seed for the classifier initialization 
            multi_class: literal indicating which multi-class scheme to use, `auto`, `multi_class` or `ovr`. See sklearn docs
            verbose: integer controlling verbosity of classifier. Set to 0 for no messages. 

        Returns:
            LogisticRegressionCV: regressor object to features, labels

        See also: 
            train_svm, sklearn.linear_model.LogisticRegressionCV
    """
    
    Cs = np.logspace(-5, 5, num_c)
    kf = KFold(n_splits=int(len(labels) / float(K)))

    n_labels = len(np.unique(labels))

    params = {
        'cv': kf,
        'random_state': seed,
        'dual': False,
        'solver': 'lbfgs',
        'class_weight': 'balanced',
        'multi_class': multi_class,
        'refit': True,
        'scoring': 'accuracy',
        'tol': 1e-6,
        'max_iter': 2000,
        'verbose': verbose
    }
    # Load and train classifier
    if penalty != 'none':
        params.update({
            'Cs': Cs,
            'penalty': penalty
        })

    return LogisticRegressionCV(**params).fit(features, labels)

def train_svm(features, labels, K: int, penalty: Penalty, num_c: int, seed: int, verbose: int=0):
    """ Trains a Kfold cross-validated SVM and returns fitted classifier

        Args:
            features: a sample x feature floating point numpy array of labeled data to be classified
            labels: an iterable of integer labels for the samples
            K: integer number of splits for cross validation
            penalty: literal indicating which sort of regularization to use, `l1`, `l2` or `elasticnet``
            num_c: integer number of regularizer constants to search over, logarithmically spaced between 1e-5 and 1e5
            seed: integer random seed for the classifier initialization 
            verbose: integer controlling verbosity of classifier. Set to 0 for no messages. 

        Returns:
            GridSearchCV: abstract cross-validation object with SVM sub- object to features, labels

        See also: 
            train_regressor, sklearn.svm.SVC
    """
 

    min_exemplars = min([len([lb for lb in labels if lb == l]) for l in np.unique(labels)])
    Cs = np.logspace(-5, 5, num_c)
    kernels=['rbf','linear']
    param_grid = {'C':Cs, 'kernel':kernels}
    kf = min(int(len(labels) / float(K)), min_exemplars)

    svc_params = {
        'class_weight': 'balanced',
        'tol': 1e-6,
        'max_iter': 2000,
        'probability':True,
        'verbose': verbose
    }
    gs_params = {'cv':kf,
                 'refit':True,
                 'scoring':'accuracy',
                 'verbose':verbose
    }

    return GridSearchCV(SVC(**svc_params), param_grid,**gs_params).fit(features,labels)
