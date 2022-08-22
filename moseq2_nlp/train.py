import os
import time
from typing import List, Literal

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report

from moseq2_nlp.data import get_embedding_representation, get_transition_representation, get_usage_representation, load_groups
from moseq2_nlp.utils import ensure_dir, write_yaml
import pdb

Representation = Literal['embeddings', 'usages', 'transitions']
Classifier = Literal['logistic_regression', 'svm']
Penalty = Literal['l1', 'l2', 'elasticnet']

def train(name: str, save_dir: str, model_path: str, index_path: str, train_inds: list, representation: Representation, classifier: Classifier, emissions: bool, custom_groupings: List[str],
          num_syllables: int, num_transitions: int, min_count: int, dm: Literal[0,1,2], embedding_dim: int, embedding_window: int,
          embedding_epochs: int, bad_syllables: List[int], K: int, penalty: Penalty, num_c: int, multi_class: str, kernel: str, seed:int):

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
    X_train = features[train_inds]
    X_test  = [feature for f, feature in enumerate(features) if f not in train_inds]
    y_train = [labels[ind] for ind in train_inds]
    y_test  = [label for l, label in enumerate(labels) if l not in train_inds]
    times['Features'] = time.time() - start

    start = time.time()
    print('Training classifier')
    if classifier == 'logistic_regression':
        # Train logistic regressor and CV over C using validation data from the training set.
        clf = train_regressor(X_train, y_train, K, penalty, num_c, seed, multi_class)
        # TODO: These are Cs per class. But what C is used at test time?
        #C = clf.C_
        # Final prediction on test set.
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
    elif classifier == 'svm':
        clf = train_svm(X_train, y_train, kernel, K, scoring, penalty, num_c, seed)
    
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
        'scoring': 'f1',
        'tol': 1e-6,
        'max_iter': 2000
    }
    # Load and train classifier
    if penalty != 'none':
        params.update({
            'Cs': Cs,
            'penalty': penalty
        })

    clf = LogisticRegressionCV(**params).fit(features, labels)

    return clf

def train_svm(features, labels, kernel: str, K: int, scoring: str, penalty: Penalty, num_c: int, seed: int):
    
    Cs = np.logspace(-5, 5, num_c)
    kf = n_splits=int(len(labels) / float(K))
    all_scores_list = []

    params = {
        'kernel': kernel,
        'random_state': seed,
        'class_weight': 'balanced',
        'tol': 1e-6,
        'max_iter': 2000
    }

    for C in Cs:
        # Load and train classifier
        params.update({
            'C': C,
        })

        clf = SVC(**params)
        scores = cross_val_score(clf, features, labels, cv=kf, scoring=scoring)
        all_scores_list.append(scores)

    all_scores = np.array(all_scores_list) # nm_C x nm_classes
    best_C_ind = np.argmax(all_scores.mean(1))
    best_C     = Cs[best_C_ind]
    best_score = all_scores[best_C_ind].mean()

    return best_C, best_score
