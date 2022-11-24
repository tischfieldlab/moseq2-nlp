import os
import time
from typing import List, Literal, Union

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from moseq2_nlp.data import get_embedding_representation, get_transition_representation, get_usage_representation, load_groups, get_emissions
from moseq2_nlp.utils import ensure_dir, write_yaml
import pickle
import pandas as pd
import pdb

import warnings
warnings.filterwarnings("ignore")

Representation = Literal['embeddings', 'usages', 'transitions']
Classifier = Literal['logistic_regression', 'svm']
Penalty = Literal['l1', 'l2', 'elasticnet']

def train(name: str, save_dir: str, data_path: str, representation: Representation, classifier: Classifier, emissions: bool, custom_groupings: List[str],
        num_syllables: int, num_transitions: int, min_count: int, negative: int, dm: Literal[0,1,2], embedding_dim: int, embedding_window: int,
          embedding_epochs: int, bad_syllables: List[int], test_size: float, K: int, penalty: Penalty, num_c: int, multi_class: str, kernel: str, seed:int, split_seed:int=None, verbose:int=0):

    np.random.seed(seed)    
    save_dict = {}
    times = {'Preamble': 0.0, 'Data': 0.0, 'Features': 0.0, 'Classifier': 0.0}

    start = time.time()
    bad_syllables = [int(bs) for bs in bad_syllables]
    exp_dir = ensure_dir(os.path.join(save_dir, name))

    times['Preamble'] = time.time() - start

    # Load data
    print(data_path)
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

    unique_labels = []
    for lb in labels:
        if lb not in unique_labels:
            unique_labels.append(lb)
    print(unique_labels)

    # Make train/test splits
    print(test_size)
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
        df.set_axis([ul + '_pred' for ul in unique_labels], axis=0, inplace=True)
        df.set_axis([ul + '_true' for ul in unique_labels], axis=1, inplace=True)
        df
        df.to_pickle(os.path.join(exp_dir, f'cm_{nm}.pkl'))

    np.save(os.path.join(exp_dir, 'features.npy'), features)

def train_regressor(features, labels, K: int, penalty: Penalty, num_c: int, seed: int, multi_class: Literal['auto', 'multi_class', 'ovr'], verbose: int=0):
    
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
