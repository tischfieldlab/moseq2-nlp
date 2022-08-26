import os
import time
from typing import List, Literal, Union

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from moseq2_nlp.data import get_embedding_representation, get_transition_representation, get_usage_representation, load_groups
from moseq2_nlp.utils import ensure_dir, write_yaml
import pdb

import warnings
warnings.filterwarnings("ignore")

Representation = Literal['embeddings', 'usages', 'transitions']
Classifier = Literal['logistic_regression', 'svm']
Penalty = Literal['l1', 'l2', 'elasticnet']

def train(name: str, save_dir: str, model_path: str, index_path: str, representation: Representation, classifier: Classifier, emissions: bool, custom_groupings: List[str],
          num_syllables: int, num_transitions: int, min_count: int, dm: Literal[0,1,2], embedding_dim: int, embedding_window: int,
          embedding_epochs: int, bad_syllables: List[int], test_size: float, K: int, penalty: Penalty, num_c: int, multi_class: str, kernel: str, seed:int, split_seed:int=None):

    np.random.seed(seed)    
    save_dict = {'parameters': locals()}
    times = {'Preamble': 0.0, 'Data': 0.0, 'Features': 0.0, 'Classifier': 0.0}

    start = time.time()

    group_map = load_groups(index_path, custom_groupings)
    bad_syllables = [int(bs) for bs in bad_syllables]
    exp_dir = ensure_dir(os.path.join(save_dir, name))

    times['Preamble'] = time.time() - start

    start = time.time()
    print('Getting features')
    if representation == 'embeddings':
        labels, features = get_embedding_representation(model_path, index_path, group_map, emissions=emissions, bad_syllables=bad_syllables,
                            dm=dm, embedding_dim=embedding_dim, embedding_window=embedding_window, embedding_epochs=embedding_epochs, min_count=min_count,
                            model_dest=os.path.join(exp_dir, 'doc2vec'), ablation='none', phrase_path=None, seed=seed)

    elif representation == 'usages':
        labels, features = get_usage_representation(model_path, index_path, group_map, num_syllables)

    elif representation == 'transitions':
        labels, features = get_transition_representation(model_path, index_path, group_map, num_transitions, max_syllable=num_syllables)

    else:
        raise ValueError('Representation type not recognized. Valid values are "usages", "transitions" and "embeddings".')

    # Make train/test splits
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=split_seed, stratify=labels)

    times['Features'] = time.time() - start

    start = time.time()
    print('Training classifier')
    if classifier == 'logistic_regressor':
        # Train logistic regressor and CV over C using validation data from the training set.
        clf = train_regressor(X_train, y_train, K, penalty, num_c, seed, multi_class)
    elif classifier == 'svm':
        clf = train_svm(X_train, y_train, K, penalty, num_c, seed)
    else:
        raise ValueError(f'Classifier {classifier} not recognized')
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    times['Classifier'] = time.time() - start

#    print(f'Best C: {C}')
    print(report)

    save_dict['model_performance'] = {
        f'classification_report': report
    }

    save_dict['compute_times'] = times
    write_yaml(os.path.join(exp_dir, 'experiment_info.yaml'), save_dict)

    #np.save(os.path.join(exp_dir, 'best_C.npy'), C)

def train_regressor(features, labels, K: int, penalty: Penalty, num_c: int, seed: int, multi_class: Literal['auto', 'multi_class', 'ovr']):
    
    Cs = np.logspace(-5, 5, num_c)
    kf = KFold(n_splits=int(len(labels) / float(K)))

    n_labels = len(np.unique(labels))
    #label_binarizer = LabelBinarizer().fit(np.arange(n_labels))

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
        'max_iter': 2000
    }
    # Load and train classifier
    if penalty != 'none':
        params.update({
            'Cs': Cs,
            'penalty': penalty
        })

    return LogisticRegressionCV(**params).fit(features, labels)

def train_svm(features, labels, K: int, penalty: Penalty, num_c: int, seed: int):

    min_exemplars = min([len([lb for lb in labels if lb == l]) for l in np.unique(labels)])
    Cs = np.logspace(-5, 5, num_c)
    kernels=['rbf','linear']
    param_grid = {'C':Cs, 'kernel':kernels}
    kf = min(int(len(labels) / float(K)), min_exemplars)

    svc_params = {
        'class_weight': 'balanced',
        'tol': 1e-6,
        'max_iter': 2000,
        'probability':True
    }
    gs_params = {'cv':kf,
                 'refit':True,
                 'scoring':'accuracy'
    }

    return GridSearchCV(SVC(**svc_params), param_grid,**gs_params).fit(features,labels)
